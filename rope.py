import math
import torch
from torchfeather.model.model_args import DeepSeekV3ModelArgs


def precompute_freqs_cls(args: DeepSeekV3ModelArgs) -> torch.Tensor:

    # ============================================================
    # 1. FREQUENCY
    # ============================================================
    #
    # Mathematical formula:
    #
    #     omega_i = theta^(-2i/d)
    #
    # where:
    #
    #     i     = 0, 1, ..., d/2 - 1
    #     d     = RoPE dimension
    #     theta = RoPE base
    #
    # Each 2D dimension pair gets its own frequency:
    #
    #     i = 0 -> (x_0, x_1)
    #     i = 1 -> (x_2, x_3)
    #     i = 2 -> (x_4, x_5)
    #     ...
    #
    dim = args.qk_rope_head_dim

    # L = sequence length
    #
    # Positions will be:
    #
    #     p in {0, 1, ..., L-1}
    #
    seqlen = args.max_seq_len

    # YaRN parameters.
   
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    factor = args.rope_factor

    # theta = RoPE base
    #
    # Mathematical notation:
    #     theta = base
    base = args.rope_theta


    # ------------------------------------------------------------
    # Calculate omega_i
    # ------------------------------------------------------------
    #
    # Code:
    #
    #     torch.arange(0, dim, 2)
    #
    # gives:
    #
    #     0, 2, 4, ..., d-2
    #
    # which is:
    #
    #     2i
    #
    # Therefore:
    #
    #     torch.arange(0, dim, 2) / dim
    #
    # corresponds to:
    #
    #     2i/d
    #
    # Then:
    #
    #     base ** (2i/d)
    #
    # gives:
    #
    #     theta^(2i/d)
    #
    # Taking reciprocal:
    #
    #     1 / theta^(2i/d)
    #
    # gives:
    #
    #     theta^(-2i/d)
    #
    # Therefore:
    #
    #     freqs[i] = omega_i
    #
    #     freqs[i] = theta^(-2i/d)
    #
    # Shape:
    #
    #     [d/2]
    #
    freqs = 1.0 / (
        base ** (
            torch.arange(0, dim, 2)[: (dim // 2)].float() / dim
        )
    )


    # 2. POSITION
    # Mathematical definition:
    #
    #     p in {0, 1, ..., L-1}
    #
    # Code:
    #
    #     t = [0, 1, ..., L-1]
    #
    # Therefore:
    #
    #     t[p] = p
    #
    t = torch.arange(seqlen)


    # ============================================================
    # 3. ANGLE
    # ============================================================
    #
    # RoPE angle:
    #
    #     phi_(p,i) = p * omega_i
    #
    # Since:
    #
    #     omega_i = theta^(-2i/d)
    #
    # we get:
    #
    #     phi_(p,i)
    #         = p * theta^(-2i/d)
    #
    # The outer product:
    #
    #     torch.outer(t, freqs)
    #
    # computes every:
    #
    #     t[p] * freqs[i]
    #
    # therefore:
    #
    #     freqs[p,i]
    #         = p * omega_i
    #
    #         = phi_(p,i)
    #
    # Shape:
    #
    #     [L, d/2]
    #
    # Rows    -> positions p
    # Columns -> frequency indices i
    #
    freqs = torch.outer(t, freqs)


    # ============================================================
    # 4. COMPLEX ROTATION
    # ============================================================
    #
    # Euler's formula:
    #
    #     e^(i phi) = cos(phi) + i sin(phi)
    #
    # Here:
    #
    #     phi = phi_(p,i)
    #
    # Therefore:
    #
    #     e^(i phi_(p,i))
    #         = cos(phi_(p,i))
    #           + i sin(phi_(p,i))
    #
    # torch.polar(radius, angle) creates:
    #
    #     radius * e^(i * angle)
    #
    # Since the radius is 1:
    #
    #     freqs_cis[p,i]
    #         = e^(i phi_(p,i))
    #
    #         = cos(phi_(p,i))
    #           + i sin(phi_(p,i))
    #
    # This complex number represents the same 2D rotation as:
    #
    #     [ cos(phi)  -sin(phi) ]
    #     [ sin(phi)   cos(phi) ]
    #
    freqs_cis = torch.polar(
        torch.ones_like(freqs),
        freqs
    )

    # Final shape:
    #
    #     [L, d/2]
    #
    return freqs_cis




def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor
) -> torch.Tensor:

    # ============================================================
    # INPUT
    # ============================================================
    #
    # x shape:
    #
    #     [B, S, H, D]
    #
    # where:
    #
    #     B = batch size
    #     S = sequence length
    #     H = number of heads
    #     D = head dimension
    #
    # RoPE works on pairs:
    #
    #     (x_0, x_1)
    #     (x_2, x_3)
    #     ...
    #
    # so D must be even.

    dtype = x.dtype


    # ============================================================
    # 5. REPRESENT INPUT PAIR AS COMPLEX
    # ============================================================
    #
    # First group the final D dimensions into pairs:
    #
    #     [B, S, H, D]
    #
    # becomes:
    #
    #     [B, S, H, D/2, 2]
    #
    # where each pair is:
    #
    #     (x_(2i), x_(2i+1))
    #
    x = x.float().view(
        *x.shape[:-1],
        -1,
        2
    )


    # Each pair:
    #
    #     (x_(2i), x_(2i+1))
    #
    # is represented as one complex number:
    #
    #     z_i = x_(2i) + i*x_(2i+1)
    #
    # Therefore:
    #
    #     real part = x_(2i)
    #     imaginary part = x_(2i+1)
    #
    # Shape:
    #
    #     [B, S, H, D/2]
    #
    x = torch.view_as_complex(x)


    # ============================================================
    # 6. MATCH POSITION + FREQUENCY DIMENSIONS
    # ============================================================
    #
    # freqs_cis originally:
    #
    #     [S, D/2]
    #
    # We reshape it to:
    #
    #     [1, S, 1, D/2]
    #
    # so it broadcasts against:
    #
    #     x -> [B, S, H, D/2]
    #
    # Thus each:
    #
    #     position p
    #
    # and each:
    #
    #     frequency i
    #
    # gets the correct:
    #
    #     e^(i phi_(p,i))
    #
    freqs_cis = freqs_cis.view(
        1,
        x.size(1),
        1,
        x.size(-1)
    )


    # ============================================================
    # 7. ROTATE
    # ============================================================
    #
    # We have:
    #
    #     z_i = x_(2i) + i*x_(2i+1)
    #
    # and:
    #
    #     e^(i phi_(p,i))
    #
    # where:
    #
    #     phi_(p,i) = p * theta^(-2i/d)
    #
    # Therefore complex multiplication gives:
    #
    #     z'_i
    #       = z_i * e^(i phi_(p,i))
    #
    # i.e.
    #
    #     z'_i
    #       = (x_(2i) + i*x_(2i+1))
    #         e^(i phi_(p,i))
    #
    y = x * freqs_cis


    # ============================================================
    # 8. EXPAND
    # ============================================================
    #
    # Using:
    #
    #     e^(i phi)
    #       = cos(phi) + i sin(phi)
    #
    # we get:
    #
    #     z'_i
    #       = (x_(2i) + i*x_(2i+1))
    #         (cos(phi) + i sin(phi))
    #
    # Expand:
    #
    #     z'_i
    #       =
    #       x_(2i) cos(phi)
    #       + i*x_(2i) sin(phi)
    #       + i*x_(2i+1) cos(phi)
    #       + i^2*x_(2i+1) sin(phi)
    #
    # Since:
    #
    #     i^2 = -1
    #
    # we get:
    #
    #     z'_i
    #       =
    #       [x_(2i) cos(phi)
    #        - x_(2i+1) sin(phi)]
    #
    #       + i[
    #          x_(2i) sin(phi)
    #          + x_(2i+1) cos(phi)
    #         ]
    #
    # Therefore:
    #
    #     x'_(2i)
    #       =
    #       x_(2i) cos(phi)
    #       - x_(2i+1) sin(phi)
    #
    #     x'_(2i+1)
    #       =
    #       x_(2i) sin(phi)
    #       + x_(2i+1) cos(phi)
    #
    # This is exactly the standard RoPE rotation matrix:
    #
    #     [ x'_(2i)   ]   [ cos(phi)  -sin(phi) ] [ x_(2i)   ]
    #     [ x'_(2i+1) ] = [ sin(phi)   cos(phi) ] [ x_(2i+1) ]
    #
    # Therefore:
    #
    #     complex multiplication
    #
    # is mathematically identical to:
    #
    #     2D RoPE rotation matrix


    # ============================================================
    # 9. CONVERT BACK TO REAL VALUES
    # ============================================================
    #
    # After rotation:
    #
    #     z'_i = a + i*b
    #
    # convert it back to:
    #
    #     (a, b)
    #
    # which gives:
    #
    #     [B, S, H, D/2, 2]
    #
    y = torch.view_as_real(y)


    # ============================================================
    # 10. RESTORE ORIGINAL D DIMENSION
    # ============================================================
    #
    # Combine:
    #
    #     D/2 pairs × 2 values
    #
    # back into:
    #
    #     D values
    #
    # Shape:
    #
    #     [B, S, H, D/2, 2]
    #
    #         ->
    #
    #     [B, S, H, D]
    #
    y = y.flatten(3)


    return y.to(dtype)


# ================================================================
# COMPLETE MATHEMATICAL PIPELINE
# ================================================================
#
# Frequency:
#
#     omega_i = theta^(-2i/d)
#
# Position:
#
#     p in {0, 1, ..., L-1}
#
# Angle:
#
#     phi_(p,i) = p * omega_i
#
#                  = p * theta^(-2i/d)
#
# Complex rotation:
#
#     e^(i phi_(p,i))
#         = cos(phi_(p,i))
#           + i sin(phi_(p,i))
#
# Represent input pair as complex:
#
#     z_i = x_(2i) + i*x_(2i+1)
#
# Rotate:
#
#     z'_i = z_i * e^(i phi_(p,i))
#
# Expand:
#
#     z'_i
#       =
#       (x_(2i) cos(phi_(p,i))
#        - x_(2i+1) sin(phi_(p,i)))
#
#       + i(
#           x_(2i) sin(phi_(p,i))
#           + x_(2i+1) cos(phi_(p,i))
#         )
#
# Therefore:
#
#     x'_(2i)
#       = x_(2i) cos(phi_(p,i))
#         - x_(2i+1) sin(phi_(p,i))
#
#     x'_(2i+1)
#       = x_(2i) sin(phi_(p,i))
#         + x_(2i+1) cos(phi_(p,i))
#
# which is exactly:
#
#     [ x'_(2i)   ]   [ cos(phi)  -sin(phi) ] [ x_(2i)   ]
#     [ x'_(2i+1) ] = [ sin(phi)   cos(phi) ] [ x_(2i+1) ]
#
#
# ================================================================
# IMPORTANT: THIS CODE IS NOT ACTUALLY APPLYING YaRN
# ================================================================
#
# Although the code reads:
#
#     beta_fast
#     beta_slow
#     factor
#
# none of them are used in the frequency calculation.
#
# Therefore the actual frequency is only:
#
#     omega_i = theta^(-2i/d)
#
# and the actual angle is:
#
#     phi_(p,i) = p * theta^(-2i/d)
#
# For YaRN, the frequency would instead become:
#
#     omega_i^YaRN
#       =
#       omega_i
#       (
#         (1-r_i)/f + r_i
#       )
#
# and therefore:
#
#     phi_(p,i)^YaRN
#       =
#       p * omega_i^YaRN
#
#       =
#       p * theta^(-2i/d)
#       (
#         (1-r_i)/f + r_i
#       )
#
# where:
#
#     r_i =
#       clip(
#         (i - i_fast) /
#         (i_slow - i_fast),
#         0,
#         1
#       )
#
# So:
#
#     YaRN changes the frequency/angle construction.
#
#     The complex rotation:
#
#         z'_i = z_i * e^(i phi)
#
#     can remain exactly the same.
#
# ================================================================