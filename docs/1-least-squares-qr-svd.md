# Least Squares via QR/SVD trong Linfa

Tài liệu này giải thích cách giải bài toán Least Squares (bình phương tối thiểu) sử dụng QR Decomposition và SVD (Singular Value Decomposition) trong thư viện Linfa.

---

## Mục lục

1. [Bài toán Least Squares](#1-bài-toán-least-squares)
2. [Các khái niệm cơ bản](#2-các-khái-niệm-cơ-bản)
3. [QR Decomposition](#3-qr-decomposition)
4. [SVD - Singular Value Decomposition](#4-svd---singular-value-decomposition)
5. [So sánh QR vs SVD](#5-so-sánh-qr-vs-svd)
6. [Implementation trong Linfa](#6-implementation-trong-linfa)
7. [Ví dụ thực tế](#7-ví-dụ-thực-tế)

---

## 1. Bài toán Least Squares

### 1.1 Định nghĩa

Cho ma trận **A** (m × n) và vector **b** (m × 1), tìm vector **x** (n × 1) sao cho:

```
minimize ||Ax - b||²
```

Trong đó `||v||²` là **squared L2 norm** (tổng bình phương các phần tử):

```
||v||² = v₁² + v₂² + ... + vₙ²
```

### 1.2 Ý nghĩa trong Linear Regression

| Ký hiệu | Tên gọi | Kích thước | Ví dụ |
|---------|---------|------------|-------|
| **A** | Feature matrix (X) | (n_samples × n_features) | Diện tích, số phòng |
| **x** | Weights/Parameters (w) | (n_features × 1) | Trọng số cần tìm |
| **b** | Target vector (y) | (n_samples × 1) | Giá nhà |

**Mô hình:**
```
ŷ = A × x = X × w

Mục tiêu: Tìm w sao cho ŷ ≈ y (dự đoán gần với thực tế)
```

### 1.3 Ví dụ minh họa

```
Dự đoán giá nhà dựa trên diện tích và số phòng:

A (features):              x (weights):      b (targets):
┌─────────────────────┐    ┌────────┐        ┌─────────┐
│ diện_tích  số_phòng │    │  w₁    │        │ giá_nhà │
├─────────────────────┤    │  w₂    │        ├─────────┤
│    50        2      │    └────────┘        │  500tr  │
│    80        3      │         ↑            │  800tr  │
│   120        4      │    cần tìm           │ 1200tr  │
└─────────────────────┘                      └─────────┘

Least Squares tìm w₁, w₂ sao cho:
  50×w₁ + 2×w₂ ≈ 500
  80×w₁ + 3×w₂ ≈ 800
 120×w₁ + 4×w₂ ≈ 1200
```

### 1.4 Tại sao không dùng Normal Equations?

**Normal Equations** (cách giải trực tiếp):
```
AᵀA × x = Aᵀb
x = (AᵀA)⁻¹ × Aᵀb
```

**Vấn đề:**
- Nhân `AᵀA` làm **bình phương condition number**: `κ(AᵀA) = κ(A)²`
- Nếu `κ(A) = 10³` → `κ(AᵀA) = 10⁶` → mất 6 chữ số thập phân!
- Ma trận `AᵀA` có thể singular (không khả nghịch)

**Giải pháp:** Dùng QR hoặc SVD decomposition.

---

## 2. Các khái niệm cơ bản

### 2.1 BLAS (Basic Linear Algebra Subprograms)

Thư viện chuẩn công nghiệp cho các phép toán đại số tuyến tính cơ bản:

| Level | Phép toán | Ví dụ |
|-------|-----------|-------|
| 1 | Vector-Vector | `y = αx + y` (axpy) |
| 2 | Matrix-Vector | `y = Ax` |
| 3 | Matrix-Matrix | `C = AB` |

**Implementations phổ biến:** OpenBLAS, Intel MKL, Apple Accelerate

### 2.2 LAPACK (Linear Algebra Package)

Xây dựng trên BLAS, cung cấp các thuật toán cấp cao:
- Giải hệ phương trình tuyến tính
- QR decomposition
- SVD decomposition
- Eigenvalue problems

```
Mối quan hệ:

┌─────────────────────────────┐
│         LAPACK              │  ← Thuật toán cấp cao
│  (QR, SVD, Eigenvalues)     │
├─────────────────────────────┤
│          BLAS               │  ← Phép tính cơ bản
│  (nhân ma trận, cộng vector)│
└─────────────────────────────┘
```

### 2.3 Ma trận chuyển vị (Transpose)

```
      ┌           ┐           ┌           ┐
      │ a  b  c   │           │ a  d  g   │
A  =  │ d  e  f   │    Aᵀ =   │ b  e  h   │
      │ g  h  i   │           │ c  f  i   │
      └           ┘           └           ┘

Quy tắc: (Aᵀ)ᵢⱼ = Aⱼᵢ (hàng ↔ cột)
```

### 2.4 Ma trận trực giao (Orthogonal Matrix)

Ma trận **Q** được gọi là trực giao nếu:
```
Qᵀ × Q = Q × Qᵀ = I (ma trận đơn vị)

Hay: Qᵀ = Q⁻¹ (chuyển vị = nghịch đảo)
```

**Tính chất quan trọng:**
- Các cột của Q trực giao với nhau (tích vô hướng = 0)
- Độ dài mỗi cột = 1 (unit vectors)
- Bảo toàn độ dài: `||Qx|| = ||x||`

---

## 3. QR Decomposition

### 3.1 Định nghĩa

Phân tách ma trận **A** (m × n) thành tích của hai ma trận:

```
A = Q × R

Trong đó:
- Q (m × m): Ma trận trực giao
- R (m × n): Ma trận tam giác trên (upper triangular)
```

**Hình dạng:**
```
┌─────────┐   ┌─────────┐   ┌─────────┐
│         │   │         │   │ × × × × │
│    A    │ = │    Q    │ × │ 0 × × × │
│  (m×n)  │   │  (m×m)  │   │ 0 0 × × │
│         │   │         │   │ 0 0 0 × │
└─────────┘   └─────────┘   └─────────┘
                               R (m×n)
```

### 3.2 Giải Least Squares bằng QR

**Bước 1:** Phân tách A = QR

**Bước 2:** Thay vào bài toán
```
||Ax - b||² = ||QRx - b||²
```

**Bước 3:** Nhân với Qᵀ (không đổi norm vì Q trực giao)
```
= ||Qᵀ(QRx - b)||²
= ||QᵀQRx - Qᵀb||²
= ||Rx - Qᵀb||²        (vì QᵀQ = I)
```

**Bước 4:** Đặt c = Qᵀb, giải hệ tam giác trên
```
Rx = c
```

### 3.3 Back-substitution (Thế ngược)

Vì R là ma trận tam giác trên, ta giải từ dưới lên:

```
    ┌                     ┐   ┌    ┐     ┌    ┐
    │ r₁₁  r₁₂  r₁₃  r₁₄ │   │ x₁ │     │ c₁ │
    │  0   r₂₂  r₂₃  r₂₄ │   │ x₂ │     │ c₂ │
R = │  0    0   r₃₃  r₃₄ │ × │ x₃ │  =  │ c₃ │
    │  0    0    0   r₄₄ │   │ x₄ │     │ c₄ │
    └                     ┘   └    ┘     └    ┘
```

**Giải:**
```
x₄ = c₄ / r₄₄

x₃ = (c₃ - r₃₄×x₄) / r₃₃

x₂ = (c₂ - r₂₃×x₃ - r₂₄×x₄) / r₂₂

x₁ = (c₁ - r₁₂×x₂ - r₁₃×x₃ - r₁₄×x₄) / r₁₁
```

**Công thức tổng quát:**
```
         cᵢ - Σⱼ₌ᵢ₊₁ⁿ (rᵢⱼ × xⱼ)
xᵢ  =  ─────────────────────────
                 rᵢᵢ

(giải từ i = n xuống i = 1)
```

**Độ phức tạp:** O(n²) - nhanh hơn nhiều so với O(n³) của Gaussian elimination

### 3.4 Thuật toán tính QR

**Gram-Schmidt Process:**
```
Cho các cột của A: a₁, a₂, ..., aₙ

q₁ = a₁ / ||a₁||

q₂ = (a₂ - (a₂·q₁)q₁) / ||...||

q₃ = (a₃ - (a₃·q₁)q₁ - (a₃·q₂)q₂) / ||...||

...
```

**Householder Reflections:** (ổn định số học hơn)
- Dùng các phép biến đổi Householder để triệt tiêu các phần tử dưới đường chéo
- Đây là phương pháp được LAPACK sử dụng

### 3.5 Điều kiện để QR hoạt động

| Điều kiện | Ý nghĩa |
|-----------|---------|
| A có **full column rank** | Các cột độc lập tuyến tính |
| m ≥ n (overdetermined) | Số mẫu ≥ số features |

**Trường hợp thất bại:**
- Các cột phụ thuộc tuyến tính → R có phần tử đường chéo = 0
- Không thể thực hiện back-substitution (chia cho 0)

---

## 4. SVD - Singular Value Decomposition

### 4.1 Định nghĩa

Phân tách ma trận **A** (m × n) thành tích của ba ma trận:

```
A = U × Σ × Vᵀ

Trong đó:
- U (m × m): Ma trận trực giao (left singular vectors)
- Σ (m × n): Ma trận đường chéo (singular values)
- Vᵀ (n × n): Ma trận trực giao (right singular vectors)
```

**Hình dạng:**
```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│         │   │         │   │ σ₁      │   │         │
│    A    │ = │    U    │ × │   σ₂   │ × │   Vᵀ    │
│  (m×n)  │   │  (m×m)  │   │     σ₃ │   │  (n×n)  │
│         │   │         │   │       0 │   │         │
└─────────┘   └─────────┘   └─────────┘   └─────────┘
```

**Tính chất:**
- Singular values: σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0 (r = rank của A)
- U, V là ma trận trực giao: UᵀU = I, VᵀV = I

### 4.2 Ý nghĩa hình học

SVD phân tích phép biến đổi tuyến tính A thành 3 bước:

```
         Vᵀ              Σ              U
Input ────────→ Rotate ────────→ Scale ────────→ Rotate ────────→ Output

1. Vᵀ: Xoay trong không gian input
2. Σ:  Co giãn theo các trục (singular values)
3. U:  Xoay trong không gian output
```

### 4.3 Giải Least Squares bằng SVD

**Bước 1:** Phân tách A = UΣVᵀ

**Bước 2:** Tính pseudo-inverse
```
A⁺ = V × Σ⁺ × Uᵀ

Trong đó Σ⁺ (pseudo-inverse của Σ):
┌           ┐        ┌               ┐
│ σ₁        │        │ 1/σ₁          │
│   σ₂      │   →    │     1/σ₂      │
│     σ₃    │        │        1/σ₃   │
│       0   │        │           0   │  ← không chia, giữ nguyên 0
└           ┘        └               ┘
```

**Bước 3:** Tính nghiệm
```
x = A⁺ × b = V × Σ⁺ × Uᵀ × b
```

### 4.4 Chi tiết từng bước tính toán

```
Cho A (m×n), b (m×1)

Bước 1: SVD
        A = U × Σ × Vᵀ

Bước 2: Tính Uᵀb
        c = Uᵀ × b                    (m×1) → (m×1)

Bước 3: Tính Σ⁺c (chia cho singular values)
        d = Σ⁺ × c
        dᵢ = cᵢ / σᵢ  (nếu σᵢ > threshold)
        dᵢ = 0        (nếu σᵢ ≤ threshold)

Bước 4: Tính nghiệm
        x = V × d                     (n×n) × (n×1) → (n×1)
```

### 4.5 Xử lý Rank-Deficient (Ma trận thiếu hạng)

SVD có thể xử lý trường hợp các cột phụ thuộc tuyến tính:

```
Singular values: [σ₁, σ₂, σ₃, σ₄] = [5.2, 3.1, 0.001, 0.0]
                                          ↑       ↑
                                      gần 0   bằng 0

Threshold = ε × max(m,n) × σ₁

Nếu σᵢ < threshold:
  - Coi như = 0
  - Không chia (tránh numerical instability)
  - Đặt 1/σᵢ = 0 trong Σ⁺
```

**Kết quả:** SVD cho **minimum-norm solution** (nghiệm có độ dài nhỏ nhất trong vô số nghiệm)

### 4.6 Thuật toán tính SVD

**Golub-Kahan Bidiagonalization:**
1. Biến đổi A thành dạng bidiagonal (chỉ có đường chéo chính và đường chéo phụ)
2. Áp dụng QR iteration để tìm singular values
3. Tính U và V từ các phép biến đổi

Đây là thuật toán được LAPACK sử dụng (routine `dgesdd` hoặc `dgesvd`).

---

## 5. So sánh QR vs SVD

### 5.1 Bảng so sánh

| Tiêu chí | QR | SVD |
|----------|----|----|
| **Tốc độ** | Nhanh hơn (~2x) | Chậm hơn |
| **Độ phức tạp** | O(mn²) | O(mn²) nhưng hằng số lớn hơn |
| **Độ ổn định** | Tốt | Tốt nhất |
| **Rank-deficient** | ❌ Thất bại | ✅ Xử lý được |
| **Underdetermined** | ❌ Thất bại | ✅ Cho minimum-norm solution |
| **Condition number** | κ(A) | κ(A) |
| **Thông tin thêm** | Không | Singular values, rank |

### 5.2 Khi nào dùng QR?

- Ma trận **well-conditioned** (condition number nhỏ)
- **Full column rank** (các cột độc lập)
- Cần **tốc độ** cao
- Số samples >> số features (overdetermined)

### 5.3 Khi nào dùng SVD?

- Ma trận có thể **rank-deficient** (multicollinearity)
- Cần **độ ổn định** cao nhất
- Muốn biết **singular values** (phân tích condition)
- Số samples < số features (underdetermined)
- Cần **truncated solution** (regularization)

### 5.4 Condition Number

```
Condition number κ(A) = σ_max / σ_min

κ(A) nhỏ   → bài toán well-conditioned (ổn định)
κ(A) lớn   → bài toán ill-conditioned (nhạy cảm với nhiễu)

Quy tắc: Mất log₁₀(κ) chữ số thập phân độ chính xác
```

---

## 6. Implementation trong Linfa

### 6.1 Kiến trúc

Linfa sử dụng **hai backend** tùy theo feature flag:

```
┌─────────────────────────────────────────────────────────┐
│                   Application Code                       │
│              (ols.rs, pca.rs, pls_svd.rs)               │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
┌─────────────────────┐       ┌─────────────────────┐
│   feature = "blas"  │       │  default (no blas)  │
│                     │       │                     │
│   ndarray-linalg    │       │    linfa-linalg     │
│   (LAPACK wrapper)  │       │   (pure Rust)       │
│                     │       │                     │
│   SVD-based solver  │       │   QR-based solver   │
└─────────────────────┘       └─────────────────────┘
```

### 6.2 Code trong ols.rs

```rust
// File: algorithms/linfa-linear/src/ols.rs

// Import tùy theo feature flag
#[cfg(not(feature = "blas"))]
use linfa_linalg::qr::LeastSquaresQrInto;  // QR method
#[cfg(feature = "blas")]
use ndarray_linalg::LeastSquaresSvdInto;   // SVD method

/// Giải bài toán: minimize ||Xb - y||²
fn solve_least_squares<F>(mut X: Array<F, Ix2>, mut y: Array<F, Ix1>) -> Result<Array1<F>, F>
where
    F: Float,
{
    let (X, y) = (X.view_mut(), y.view_mut());

    // Không có BLAS → dùng QR (pure Rust, nhanh)
    #[cfg(not(feature = "blas"))]
    let out = X
        .least_squares_into(y.insert_axis(Axis(1)))?
        .remove_axis(Axis(1));

    // Có BLAS → dùng SVD (LAPACK, robust hơn)
    #[cfg(feature = "blas")]
    let out = X
        .with_lapack()
        .least_squares_into(y.with_lapack())
        .map(|x| x.solution)?
        .without_lapack();

    Ok(out)
}
```

### 6.3 Fit function

```rust
// File: algorithms/linfa-linear/src/ols.rs

fn fit(&self, dataset: &DatasetBase<...>) -> Result<Self::Object, F> {
    let X = dataset.records();      // Feature matrix (A)
    let y = dataset.as_single_targets();  // Target vector (b)

    if self.fit_intercept {
        // Thêm cột 1 vào X để fit intercept
        // [X | 1] → cho phép học bias term
        let X = concatenate(Axis(1), &[X.view(), Array2::ones((X.nrows(), 1)).view()]);

        let params = solve_least_squares(X, y)?;  // [w₁, w₂, ..., wₙ, b]
        let intercept = params.last();            // b = intercept
        let params = params[..n-1];               // [w₁, w₂, ..., wₙ]

        Ok(FittedLinearRegression { intercept, params })
    } else {
        // Không fit intercept, b = 0
        let params = solve_least_squares(X, y)?;
        Ok(FittedLinearRegression {
            intercept: 0,
            params
        })
    }
}
```

### 6.4 Pseudo-inverse Implementation

```rust
// File: algorithms/linfa-pls/src/utils.rs

/// Tính pseudo-inverse bằng SVD
pub fn pinv2<F: Float>(x: ArrayView2<F>, cond: Option<F>) -> Array2<F> {
    // Bước 1: SVD decomposition
    let (u, s, vh) = x.svd(true, true).unwrap();

    // Bước 2: Tính threshold để lọc singular values nhỏ
    let cond = cond.unwrap_or(
        s.max() * max(m, n) * F::epsilon()  // machine epsilon
    );

    // Bước 3: Đếm effective rank
    let rank = s.iter().filter(|&v| v > cond).count();

    // Bước 4: Tính A⁺ = V × Σ⁺ × Uᵀ
    // Chỉ dùng 'rank' singular values đầu tiên
    let u_cut = u.slice(s![.., ..rank]) / s.slice(s![..rank]);
    vh.slice(s![..rank, ..]).t().dot(&u_cut.t())
}
```

### 6.5 Luồng xử lý tổng thể

```
┌──────────────────────────────────────────────────────────────────┐
│                     LinearRegression::fit()                       │
│                                                                  │
│  Input: X (n_samples × n_features), y (n_samples)                │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                  Thêm cột 1 nếu fit_intercept                    │
│                  X' = [X | 1]                                    │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    solve_least_squares(X', y)                     │
│                    minimize ||X'w - y||²                          │
└────────────────────────────┬─────────────────────────────────────┘
                             │
             ┌───────────────┴───────────────┐
             │                               │
             ▼                               ▼
┌────────────────────────┐       ┌────────────────────────┐
│    Có BLAS (LAPACK)    │       │   Không BLAS (Rust)    │
│                        │       │                        │
│  1. A = UΣVᵀ (SVD)     │       │  1. A = QR             │
│  2. x = VΣ⁺Uᵀb         │       │  2. c = Qᵀb            │
│                        │       │  3. Rx = c (back-sub)  │
└────────────┬───────────┘       └────────────┬───────────┘
             │                               │
             └───────────────┬───────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                    Trả về: params, intercept                      │
│                                                                  │
│  Model: y = X × params + intercept                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 7. Ví dụ thực tế

### 7.1 Ví dụ QR Decomposition

```
Bài toán: Fit đường thẳng qua 3 điểm (0,1), (1,2), (2,2)

A = ┌     ┐      b = ┌   ┐
    │ 1 0 │          │ 1 │     (cột 1 = bias, cột 2 = x)
    │ 1 1 │          │ 2 │
    │ 1 2 │          │ 2 │
    └     ┘          └   ┘

Bước 1: QR Decomposition
A = Q × R

Q = ┌                    ┐      R = ┌            ┐
    │ -0.577  0.707      │          │ -1.73  -1.73│
    │ -0.577  0.000      │          │  0      1.41│
    │ -0.577 -0.707      │          └            ┘
    └                    ┘

Bước 2: Tính c = Qᵀb
c = Qᵀ × b = ┌       ┐
             │ -2.89 │
             │ -0.71 │
             └       ┘

Bước 3: Giải Rx = c (back-substitution)
-1.73×x₁ - 1.73×x₂ = -2.89
          1.41×x₂ = -0.71

x₂ = -0.71 / 1.41 = -0.5  (slope)
x₁ = (-2.89 + 1.73×(-0.5)) / (-1.73) = 1.17  (intercept)

Kết quả: y = 1.17 - 0.5x
         (không chính xác vì 3 điểm không thẳng hàng)
```

### 7.2 Ví dụ SVD

```
Bài toán: Cùng bài toán trên

A = ┌     ┐      b = ┌   ┐
    │ 1 0 │          │ 1 │
    │ 1 1 │          │ 2 │
    │ 1 2 │          │ 2 │
    └     ┘          └   ┘

Bước 1: SVD của A
A = U × Σ × Vᵀ

U = ┌                      ┐
    │ -0.22  0.87  0.45    │
    │ -0.52  0.22 -0.82    │
    │ -0.82 -0.44  0.37    │
    └                      ┘

Σ = ┌           ┐
    │ 2.46      │
    │      0.77 │
    └           ┘

Vᵀ = ┌              ┐
     │ -0.64  -0.77 │
     │ -0.77   0.64 │
     └              ┘

Bước 2: Tính Uᵀb
Uᵀb = ┌       ┐
      │ -3.29 │
      │  0.22 │
      │ -0.45 │  ← bỏ qua (nằm ngoài column space)
      └       ┘

Bước 3: Tính Σ⁺(Uᵀb)
Σ⁺(Uᵀb) = ┌            ┐
          │ -3.29/2.46 │   ┌       ┐
          │  0.22/0.77 │ = │ -1.34 │
          └            ┘   │  0.29 │
                           └       ┘

Bước 4: Tính x = V × Σ⁺Uᵀb
x = Vᵀᵀ × ┌       ┐   ┌      ┐
          │ -1.34 │ = │ 1.17 │  ← intercept
          │  0.29 │   │ 0.50 │  ← slope
          └       ┘   └      ┘

Kết quả: y = 1.17 + 0.5x
```

### 7.3 Sử dụng trong Rust với Linfa

```rust
use linfa::prelude::*;
use linfa_linear::LinearRegression;
use ndarray::array;

fn main() {
    // Tạo dataset
    let features = array![[0.], [1.], [2.]];
    let targets = array![1., 2., 2.];
    let dataset = Dataset::new(features, targets);

    // Fit model (sử dụng QR hoặc SVD tùy feature flag)
    let model = LinearRegression::new()
        .with_intercept(true)
        .fit(&dataset)
        .unwrap();

    // Kết quả
    println!("Intercept: {}", model.intercept());  // ≈ 1.17
    println!("Slope: {:?}", model.params());       // ≈ [0.5]

    // Dự đoán
    let predictions = model.predict(&dataset);
    println!("Predictions: {:?}", predictions);    // [1.17, 1.67, 2.17]
}
```

---

## Tài liệu tham khảo

1. **Numerical Linear Algebra** - Trefethen & Bau
2. **Matrix Computations** - Golub & Van Loan
3. [LAPACK Documentation](https://www.netlib.org/lapack/)
4. [Linfa GitHub Repository](https://github.com/rust-ml/linfa)
5. [ndarray-linalg Documentation](https://docs.rs/ndarray-linalg/)

---

*Tài liệu này được tạo để giải thích cách Linfa triển khai Least Squares regression sử dụng QR Decomposition và SVD.*
