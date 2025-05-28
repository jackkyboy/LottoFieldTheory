# 🧠 Quantum-Inspired Thai Lottery Prediction – Technical Overview

> โครงการนี้จำลองกระบวนการคำนวณคล้ายกลศาสตร์ควอนตัมเพื่อนำมาวิเคราะห์และพยากรณ์ผลสลากกินแบ่งไทย โดยใช้แนวคิดฟังก์ชันคลื่น, ความน่าจะเป็นแบบ Boltzmann, และความเชื่อมโยงเชิงพันธะควอนตัม (Entanglement)

---

## 🌀 ขั้นตอนหลักของระบบ (Quantum Pipeline)

| ขั้นตอน                   | เทคนิคที่ใช้                  | เปรียบเทียบกับฟิสิกส์ควอนตัม    |
| ----------------------- | ------------------------- | ------------------------ |
| 1. แปลง Draw เป็นเวกเตอร์ | Feature Embedding → ℝ¹⁰⁰⁰ | สร้างสนามเวกเตอร์ควอนตัม    |
| 2. วัดความไม่แน่นอน        | Shannon Entropy           | วัดความซับซ้อนของสนาม       |
| 3. สร้างฟังก์ชันคลื่น         | Softmax + Temperature     | เตรียมสถานะ superposition |
| 4. ยุบสถานะ → เลขเดียว    | Weighted Collapse         | วัดระบบ → ได้ draw เดียว    |
| 5. ปรับตามบริบท           | Contextual Biasing        | Interference จากข้อมูลเสริม |

---

## 🔢 Step-by-Step พร้อมสมการ

### 1. แปลงข้อมูลเป็นเวกเตอร์ใน Quantum Field (1000 มิติ)

ใช้ข้อมูลจากรางวัลแต่ละงวด เช่น:

- `first_prize`
- `last2`, `last3`, `front3`
- วันที่, ลำดับงวด

จากนั้นแปลงเป็นเวกเตอร์ขนาด 1000 มิติด้วยฟังก์ชัน `generate_lotto_field(df, dimension=1000)`

\[
\vec{x}_i = f(\text{draw features}) \in \mathbb{R}^{1000}
\]

---

### 2. วัดความซับซ้อนด้วย Shannon Entropy

วัด entropy ของสนาม (field) เพื่อดูว่า “ข้อมูลมีลักษณะสุ่ม หรือจัดระเบียบ”

\[
H = - \sum_{i} p_i \log_2 p_i
\]

โดยที่ \( p_i \) มาจาก histogram ของค่าในเวกเตอร์

ฟังก์ชัน:  
entropy_result = entropy_test_vs_random(field)
```

### 3. สร้างฟังก์ชันคลื่นของ Draw ทั้งหมด

จากรายชื่อ Draw ทั้งหมด เช่น `["Draw_0", ..., "Draw_999"]` สร้าง superposition ด้วย softmax + temperature
$$
P(D_i) = \frac{e^{-\beta E_i}}{\sum_j e^{-\beta E_j}}, \quad \text{โดย } \beta = \frac{1}{T}
$$

- $E_i$: พลังงาน หรือความต่างใน latent space
- $T$: อุณหภูมิที่ควบคุมการกระจายความน่าจะเป็น

ฟังก์ชัน:

```

superposition = generate_schrodinger_superposition(state_ids, temperature=0.5, seed=42)
```

------

### 4. Collapse → สุ่ม Draw เดียวออกมา

เปรียบเสมือนการ "วัดสถานะควอนตัม" แล้วระบบยุบลงสู่ Draw เดียว

```


คัดลอกแก้ไข
collapsed_draw = simulate_wavefunction_collapse(superposition)
```

เบื้องหลังคือ:

```

np.random.choice(draw_ids, p=probabilities)
```

------

### 5. ปรับความน่าจะเป็นด้วยบริบท (Context Vector)

เพื่อให้การเลือก Draw คำนึงถึง "บริบท" เช่น วันใกล้เคียง, ลำดับงวด, ความเหมือนกับ Draw ที่เคยออก

ใช้ฟังก์ชัน:

```

weighted_sp = weight_superposition_with_context(superposition, context_vector, draw_metadata)
```

สมการ:
$$
P'(D_i) = \alpha \cdot P(D_i) + \gamma \cdot \text{ContextBias}(D_i)
$$

------

## 🧬 ส่วนเสริม: Embedding และ Sampling ใน Latent Space

### 🔹 Digit Embedding:

ใช้ `nn.Embedding` เพื่อแปลงเลข 0–9 แต่ละหลักใน draw เป็นเวกเตอร์ แล้วเฉลี่ยเป็น latent vector:
$$
\vec{z}_{\text{draw}} = \frac{1}{6} \sum_{k=1}^{6} \text{Embed}(d_k)
$$

### 🔹 Boltzmann Sampling ใน Latent Space:

$$
P_i = \frac{e^{-\frac{E_i}{T}}}{\sum_j e^{-\frac{E_j}{T}}}
\quad \text{โดย } E_i = \| z_i - \bar{z} \|
$$

- $z_i$: latent vector ของ draw
- $\bar{z}$: centroid ของ latent space

------

## 📊 ค่าทางสถิติที่ตรวจสอบแล้ว

- Real Entropy ≪ Random Entropy → ข้อมูลหวย "มีโครงสร้าง"
- Bell Test = 0 → ไม่พบ violation แบบ EPR paradox
- KL Divergence ต่ำ → Superposition แบบจำลองใกล้เคียงจริง

------

## 🧠 สรุป

โปรเจคนี้คือการนำกลศาสตร์ควอนตัมเชิงแนวคิด (conceptual quantum mechanics) มาใช้กับข้อมูลหวยไทย โดยเน้นการ:

- สร้างฟิลด์ข้อมูล (vector field)
- วิเคราะห์พลังงานแฝง
- จำลองการยุบตัวของฟังก์ชันคลื่น
- ใช้ context-aware interference ในการเลือกเลข