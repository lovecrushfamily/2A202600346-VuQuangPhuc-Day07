# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Vũ Quang Phúc
**Nhóm:** Nhóm 1
**Ngày:** 10/04/2026

---

## 1. Warm-up (5 điểm) 

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> *Viết 1-2 câu:* Nó có nghĩa là 2 đoạn văn bản có "ý nghĩa" (ngữ nghĩa/semantic) rất giống nhau, khi biến thành vector thì chúng trỏ về gần cùng một hướng trong không gian đa chiều (góc giữa hai vector rất nhỏ).

**Ví dụ HIGH similarity:**
- Sentence A: Khách hàng cần hỗ trợ kỹ thuật về phần mềm.
- Sentence B: Người dùng đang yêu cầu trợ giúp về lỗi ứng dụng.
- Tại sao tương đồng: Khác biệt hoàn toàn về mặt từ vựng ("khách hàng" - "người dùng", "phần mềm" - "ứng dụng") nhưng ý nghĩa ngữ cảnh tổng thể là y hệt nhau. Vector của chúng sẽ gần nhau.

**Ví dụ LOW similarity:**
- Sentence A: Khách hàng cần hỗ trợ kỹ thuật về phần mềm.
- Sentence B: Thời tiết hôm nay có mưa rào rải rác.
- Tại sao khác: Hai câu nhắc đến 2 lĩnh vực hoàn toàn không liên quan (IT vs Thời tiết). Vector của chúng sẽ vuông góc hoặc ngược hướng.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> *Viết 1-2 câu:* Khác với Euclidean (đo khoảng cách vật lý phụ thuộc vào độ dài nội dung), Cosine chỉ xét đến "Góc" (chiều hướng của ý nghĩa). Nhờ vậy, một đoạn văn dài và một đoạn văn ngắn vẫn có thể có độ tương đồng cao nếu chúng cùng nói về 1 chủ đề.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:* Công thức: `ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = 22.11`
> *Đáp án:* Làm tròn lên (ceil), ta có 23 chunks.

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> *Viết 1-2 câu:* Số chunks sẽ là `ceil((10000 - 100) / (500 - 100)) = ceil(9900 / 400) = 25`. Việc tăng Overlap giúp chia cắt an toàn hơn, tránh tình trạng một ý (context) hoặc một câu bị chặt làm đôi ở hai chunk riêng biệt làm mất nghĩa.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Chính sách công ty (Internal HR & Policy)

**Tại sao nhóm chọn domain này?**
> *Viết 2-3 câu:* Dữ liệu về chính sách và luật thường chứa các thông tin cụ thể, luật lệ có tính "đúng/sai" rõ ràng nên rất dễ để test xem mô hình RAG có trả lời chính xác hay bị ảo giác (hallucinate) hay không. Hơn nữa, nó rất thực tế để làm trợ lý nội bộ cho nhân viên.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | quy_dinh_nghi_phep.md | Sổ tay nội bộ | 3500 | `type: policy`, `department: HR` |
| 2 | che_do_bao_hiem.md | Sổ tay nội bộ | 4100 | `type: health`, `department: HR` |
| 3 | huong_dan_onboarding.txt | Training | 2800 | `type: guide`, `department: General` |
| 4 | quy_tac_bao_mat_IT.md | IT Policy | 5000 | `type: security`, `department: IT` |
| 5 | bao_cao_tai_chinh_Q1.md| Kế toán | 2000 | `type: report`, `department: Finance` |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `department` | str | `"HR"`, `"IT"` | Có thể filter theo phòng ban trước khi search, tránh việc RAG lấy luật của phòng IT sang trả lời cho kế toán. |
| `type`       | str | `"policy"`, `"guide"`| Dễ dàng lọc ra quy định khắt khe (policy) khác với các mẹo, hướng dẫn mềm (guide). |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Nội bộ | FixedSizeChunker (`fixed_size`) | 35 | 200 | Kém (Vì hay bị chặt đứt ngang câu) |
| Nội bộ | SentenceChunker (`by_sentences`) | 42 | 165 | Tốt (Giữ trọn vẹn ngữ nghĩa câu) |
| Nội bộ | RecursiveChunker (`recursive`) | 28 | 240 | Cực Rất Tốt (Giữ được đoạn mạch lạc) |

### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**
> *Viết 3-4 câu: strategy chunk thế nào? Dựa trên dấu hiệu gì?* Nó cố gắng dùng dấu phân tách lớn nhất (ví dụ: `\n\n` - giữa hai đoạn văn). Nếu đoạn văn vẫn lớn hơn chunk_size, nó đệ quy bé lại và tách tiếp bằng dấu xuống dòng (`\n`), rồi đến dấu chấm (`.`),... Tóm lại nó luôn bảo toàn cấu trúc lớn nhất có thể trước khi buộc phải xé nhỏ câu.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Viết 2-3 câu: domain có pattern gì mà strategy khai thác?* Tài liệu HR của công ty thường chia thành rất nhiều Điều, Khoản (có số thứ tự và dấu nhảy cách `\n\n` rõ ràng). Việc dùng RecursiveChunker giúp nó ôm trọn vẹn một "Điều 1, Khoản 2" vào trong một chunk thay vì băm lộn xộn.

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Nội bộ | SentenceChunker (best baseline) | 42 | 165 | Bị chia quá nát, thiếu context xung quanh. |
| Nội bộ | **RecursiveChunker (của tôi)** | 28 | 240 | Tốt nhất, Chunk chứa trọn vẹn 1 điều luật. |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi (Phúc)| RecursiveChunker | 9/10 | Giữ form Điều/Khoản luật tốt | Dễ vượt nhẹ chunk limit nếu đoạn quá dài |
| Hoàng | FixedSize (300) | 6/10 | Cắt đoạn cực nhanh, setup đơn giản | Hay bị chia cắt ngang một câu, gây mất ngữ nghĩa |
| Đôn Đức | Sentence (3 câu) | 8/10 | Đảm bảo tuyệt đối không vỡ cấu trúc câu | Thiếu bối cảnh nếu các câu trước/sau dùng đại từ thay thế (như "Anh ấy", "Nó") |
| Anh Quân | Markdown Header | 8.5/10 | Phân trọn vẹn chủ đề theo thẻ H1, H2 | Một số thẻ H2 nội dung quá dài nhét không vừa context |
| Minh Luân | FixedSize (500) | 7/10 | Lấy được đoạn dài để giữ bối cảnh tốt | Dễ dư thừa text nhiễu ở hai đầu, làm sai lệch điểm Similarity |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:* RecursiveChunker tốt nhất. Vì file Docs/Policy phụ thuộc rất nhiều vào cấu trúc đoạn paragprah (từng đoạn thường đi kèm một chủ đề đồng nhất).

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> *Viết 2-3 câu: Dùng hàm re.split() của thư viện 're' để chặt theo các pattern `\. |\! |\? |\.\n`. Gắn lại các chuỗi con đứt rải rác sau đó và ghép `max_sentences_per_chunk` câu lại với nhau thành 1 chunk trọn vẹn.*

**`RecursiveChunker.chunk` / `_split`** — approach:
> *Viết 2-3 câu: Thuật toán đệ quy. Base case: Mảnh ghép nhỏ hơn chunk_size thì nhét vào mảng. Nếu lớn hơn thì chia tiếp bằng array separator ở level thấp hơn (VD tách theo chấm câu). Dùng vòng lặp nhét dồn các mảnh này cho tới khi chạm chunk_size mới cắt sang chunk mới.*

### EmbeddingStore

**`add_documents` + `search`** — approach:
> *Viết 2-3 câu: Tôi duyệt list chứa records thành metadata dict in-memory. `search` sẽ embed ngay cái câu hỏi thành vector rồi chạy loop để tính _dot product, vứt kết quả ra mảng, sort lại và cắt lấy `top_k`.*

**`search_with_filter` + `delete_document`** — approach:
> *Viết 2-3 câu: `search_with_filter` sẽ sàng lọc danh sách vector (for match) KHỚP VỚI metadata dictionary TRƯỚC (để mảng bé đi), sau đó mới tính search _dot đắt đỏ sau (tối ưu tốc độ). Delete dùng list comprehension loại Id.*

### KnowledgeBaseAgent

**`answer`** — approach:
> *Viết 2-3 câu: Tôi gọi hàm store.search để lấy top_k các chunks. Nối nội dung các Chunk lại bằng kí tự xuốnng dòng để làm Context (bối cảnh). Sau đó nhồi Context này cùng với Câu hỏi gốc vào 1 Prompt template để giao LLM đọc.*

### Test Results

```
test_add_documents_increases_size ... ok
test_add_more_increases_further ... ok
test_chunker_classes_exist ... ok
test_chunks_are_strings ... ok
...
Ran 42 tests in 0.006s
OK
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | I love cats | Felines are my pets | high | 0.92 | True |
| 2 | I am happy | I am not sad | high | 0.88 | True |
| 3 | AI is good | Apple is red | low | 0.12 | True |
| 4 | Bank river | Bank account money | low | 0.35 | True |
| 5 | Big factory | Huge manufacturing plant| high | 0.95 | True |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> *Viết 2-3 câu:* Bất ngờ nhất là Pair 4. Cùng dùng chữ "Bank" nhưng AI hiểu "Bank river" (bờ sông) và "Bank money" (ngân hàng) là hoàn toàn khác nhau. Điều này chứng tỏ Model Embedding đã vượt qua level map keyword (khớp mồm vựng) và thực sự ĐỌC HIỂU ĐƯỢC ngữ cảnh (contextual meaning).

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **5 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Ai được hưởng bảo hiểm thai sản? | Nhân viên nữ làm việc trên 6 tháng. |
| 2 | Có được sử dụng laptop công ty cho việc cá nhân không? | Không theo điều 4 mục IT Policy. |
| 3 | Thử việc được trả bao nhiêu % lương? | Trả 85% lương chính thức. |
| 4 | Xin nghỉ phép cần báo trước mấy ngày? | Phải báo trước 2 ngày cho Quản lý. |
| 5 | Công ty nghỉ Tết âm lịch bao nhiêu ngày? | Nghỉ 7 ngày theo luật nhà nước. |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Bảo hiểm thai sản | Điều 5: Bảo hiểm. Nữ nv >6 tháng... | 0.89 | Yes | Nhân viên nữ làm >6 tháng được hưởng. |
| 2 | Note laptop cá nhân| Điều 4 Nội quy thiết bị văn phòng... | 0.91 | Yes | Không được sử dụng cho mục đích ngoài công việc. |
| 3 | Lương thử việc | Bảng lương và Chế độ đãi ngộ thử việc... | 0.85 | Yes | Được trả 85% lương cơ bản. |
| 4 | Xin nghỉ phép | Quy trình báo phép nghỉ phép năm... | 0.88 | Yes | Phải báo trước tối thiểu 2 ngày. |
| 5 | Lịch nghỉ tết | Lịch nghỉ lễ - tết nguyên đán âm lịch...| 0.95 | Yes | Nghỉ Tết 7 ngày theo nhà nước. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 5

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> *Viết 2-3 câu:* Nếu tài liệu quá lớn mà không có filter Metadata Department thì truy vấn "Nghỉ phép" rất dễ lấy nhầm cách xin phép của nhân viên part-time thay vì full-time. Sức mạnh của Metadata Filter là cực kì quan trọng.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:* Có nhóm áp dụng chunk by markdown headers (Tách đoạn chỉ bằng ký hiệu `#`, `##` của file MD). Bằng cách này các chủ đề nằm trọn trong 1 chunk cực kì logic, tốt hơn hẳn cách chặt kí tự đơn thuần.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> *Viết 2-3 câu:* Tôi sẽ làm sạch dữ liệu (Clean data) tốt hơn trước khi nhét vào RAG. Xoá bỏ các trang mục lục, footer chân trang, header lặp lại... vì chúng tạo nhiễu làm AI hiểu nhầm text dư thừa đó là nội dung chính.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |
