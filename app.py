
import os
import re
import time
import pickle
import base64
import mimetypes
from pathlib import Path

import faiss
import gradio as gr
import numpy as np
from PIL import Image
from fastembed import TextEmbedding
from google import genai
from google.genai import types


# ============================================================
# 1. CẤU HÌNH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

INDEX_PATH = BASE_DIR / "database" / "index.faiss"
METADATA_PATH = BASE_DIR / "database" / "metadata.pkl"

BG1_PATH = BASE_DIR / "assets" / "background1.jpg"
BG2_PATH = BASE_DIR / "assets" / "background2.jpg"

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "all-MiniLM-L6-v2"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "Chưa cấu hình Secret GOOGLE_API_KEY."
    )


# ============================================================
# 2. KHỞI TẠO GEMINI
# ============================================================

client = genai.Client(
    api_key=GOOGLE_API_KEY
)


# ============================================================
# 3. NẠP EMBEDDING MODEL, FAISS VÀ METADATA
# ============================================================

if not INDEX_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy: {INDEX_PATH}"
    )

if not METADATA_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy: {METADATA_PATH}"
    )

embedding_model = TextEmbedding(
    model_name=EMBEDDING_MODEL_NAME
)

index = faiss.read_index(
    str(INDEX_PATH)
)

with open(METADATA_PATH, "rb") as file:
    metadata = pickle.load(file)

if index.ntotal != len(metadata):
    raise RuntimeError(
        "Số vector FAISS không khớp metadata."
    )

if index.d != 384:
    raise RuntimeError(
        f"Dimension FAISS không đúng: {index.d}"
    )


# ============================================================
# 4. HÀM TIỆN ÍCH
# ============================================================

def file_to_base64(path: Path) -> str:
    if not path.exists():
        return ""

    with open(path, "rb") as file:
        return base64.b64encode(
            file.read()
        ).decode("utf-8")


bg1 = file_to_base64(BG1_PATH)
bg2 = file_to_base64(BG2_PATH)


def generate_content(
    contents,
    retries: int = 2
) -> str:
    """
    Gọi Gemini an toàn.

    - 429/quota: không retry liên tục
    - Lỗi tạm thời khác: retry ngắn
    - Không đưa chi tiết kỹ thuật ra giao diện
    """

    last_error = None

    for attempt in range(retries + 1):

        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents
            )

            if response.text:
                return response.text.strip()

            raise RuntimeError(
                "Gemini không trả về nội dung."
            )

        except Exception as error:

            last_error = error
            error_text = str(error).lower()

            # -----------------------------------------------
            # Hết quota / rate limit
            # -----------------------------------------------

            quota_error = (
                "429" in error_text
                or "resource_exhausted" in error_text
                or "quota exceeded" in error_text
                or "rate limit" in error_text
            )

            if quota_error:
                raise RuntimeError(
                    "Dịch vụ AI đang tạm đạt giới hạn lượt sử dụng. "
                    "Vui lòng thử lại sau vài phút."
                )

            # -----------------------------------------------
            # Lỗi khác: thử lại ngắn
            # -----------------------------------------------

            if attempt < retries:
                time.sleep(2 ** attempt)
                continue

    raise RuntimeError(
        "Dịch vụ AI tạm thời chưa phản hồi. "
        "Vui lòng thử lại sau."
    )

def normalize_text(text):
    text = str(text or "").lower()

    text = re.sub(
        r"[^\wÀ-ỹ]+",
        " ",
        text,
        flags=re.UNICODE
    )

    return " ".join(text.split())


def lexical_search(question, k=8):
    """
    Tìm kiếm từ khóa trực tiếp trong metadata.
    Giúp tăng độ chính xác với câu hỏi tiếng Việt.
    """

    query = normalize_text(question)

    words = [
        word
        for word in query.split()
        if len(word) >= 2
    ]

    if not words:
        return []

    scored = []

    for idx, item in enumerate(metadata):

        text = normalize_text(
            item.get("text", "")
        )

        score = 0

        # Khớp cả cụm câu
        if query in text:
            score += 8

        # Khớp từng từ
        for word in words:

            count = text.count(word)

            if count:
                score += min(count, 4)

        if score > 0:
            scored.append(
                (
                    score,
                    idx
                )
            )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    results = []

    for score, idx in scored[:k]:

        item = dict(metadata[idx])

        item["lexical_score"] = float(score)

        results.append(item)

    return results


def search(question: str, k: int = 5) -> list[dict]:

    if not question or not question.strip():
        return []

    # --------------------------------------------------------
    # 1. FAISS semantic search
    # --------------------------------------------------------

    query_vectors = list(
        embedding_model.query_embed(
            [question.strip()]
        )
    )

    query_vector = np.asarray(
        query_vectors,
        dtype=np.float32
    )

    faiss.normalize_L2(
        query_vector
    )

    semantic_k = min(
        max(int(k) * 3, 10),
        index.ntotal
    )

    scores, indices = index.search(
        query_vector,
        semantic_k
    )

    semantic_results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0 or idx >= len(metadata):
            continue

        item = dict(metadata[idx])

        item["score"] = float(score)

        semantic_results.append(item)


    # --------------------------------------------------------
    # 2. Keyword search
    # --------------------------------------------------------

    keyword_results = lexical_search(
        question,
        k=max(int(k) * 3, 10)
    )


    # --------------------------------------------------------
    # 3. Gộp kết quả
    # --------------------------------------------------------

    merged = {}

    def result_key(item):

        return (
            str(item.get("source", "")),
            str(item.get("page", "")),
            str(item.get("text", ""))[:120]
        )


    # Semantic
    for rank, item in enumerate(
        semantic_results
    ):

        key = result_key(item)

        combined_score = (
            1.0 / (rank + 1)
        )

        merged[key] = {
            "item": item,
            "combined_score": combined_score
        }


    # Lexical
    for rank, item in enumerate(
        keyword_results
    ):

        key = result_key(item)

        keyword_bonus = (
            1.5 / (rank + 1)
        )

        if key in merged:

            merged[key]["combined_score"] += (
                keyword_bonus
            )

        else:

            merged[key] = {
                "item": item,
                "combined_score": keyword_bonus
            }


    # --------------------------------------------------------
    # 4. Sắp xếp
    # --------------------------------------------------------

    ranked = sorted(
        merged.values(),
        key=lambda x: x["combined_score"],
        reverse=True
    )

    final_results = []

    for entry in ranked[:int(k)]:

        item = entry["item"]

        item["combined_score"] = float(
            entry["combined_score"]
        )

        final_results.append(item)

    return final_results


def build_context(
    results: list[dict],
    max_chars_per_chunk: int = 1200
) -> str:
    parts = []

    for number, item in enumerate(
        results,
        start=1
    ):
        text = item.get(
            "text",
            ""
        )[:max_chars_per_chunk]

        parts.append(
            f"""
[TÀI LIỆU {number}]
Nguồn: {item.get("source", "Không rõ")}
Trang: {item.get("page", "Không rõ")}
Nội dung:
{text}
""".strip()
        )

    return "\n\n".join(parts)


def build_sources(
    results: list[dict]
) -> str:
    sources = sorted(
        {
            f"📄 {item.get('source', 'Không rõ')} "
            f"- Trang {item.get('page', 'Không rõ')}"
            for item in results
        }
    )

    return "\n".join(sources)


# ============================================================
# 6. RAG VĂN BẢN
# ============================================================


def ask_rag_with_source(
    question: str,
    k: int = 6
):
    """
    RAG văn bản:
    - Hybrid Search tìm các đoạn liên quan
    - Gemini tổng hợp thông tin từ nhiều đoạn
    - Không bịa ngoài tài liệu
    """

    if not question or not question.strip():
        return "Bạn chưa nhập câu hỏi.", ""

    # --------------------------------------------------------
    # 1. Tìm tài liệu
    # --------------------------------------------------------

    results = search(
        question,
        k=k
    )

    if not results:
        return (
            "Không tìm thấy tài liệu phù hợp với câu hỏi.",
            ""
        )

    # --------------------------------------------------------
    # 2. Tạo context
    # --------------------------------------------------------

    context = build_context(
        results,
        max_chars_per_chunk=1800
    )

    if not context.strip():
        return (
            "Không tìm thấy nội dung phù hợp trong tài liệu.",
            build_sources(results)
        )

    # --------------------------------------------------------
    # 3. Prompt RAG cải tiến
    # --------------------------------------------------------

    prompt = f"""
Bạn là ANGriTECH, trợ lý AI hỗ trợ sản xuất
nông nghiệp tại Đắk Lắk.

CÂU HỎI CỦA NGƯỜI DÙNG:
{question}

TÀI LIỆU TRUY XUẤT:
{context}

NHIỆM VỤ:

1. Hãy đọc TẤT CẢ các đoạn tài liệu được cung cấp.

2. Nếu thông tin cần thiết nằm rải rác ở nhiều đoạn,
   hãy tổng hợp chúng thành một câu trả lời thống nhất.

3. Nếu tài liệu chỉ trả lời được một phần câu hỏi,
   vẫn phải trả lời phần có căn cứ và nói rõ phần nào
   tài liệu chưa cung cấp đầy đủ.

4. KHÔNG được từ chối trả lời chỉ vì không có một đoạn
   duy nhất chứa toàn bộ đáp án.

5. Chỉ được sử dụng thông tin có trong tài liệu.
   Không tự bổ sung kiến thức bên ngoài.

6. Không tự đề xuất thuốc, hóa chất, liều lượng hoặc
   quy trình xử lý nếu tài liệu không nêu.

7. Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu.

8. Nếu tài liệu có các mốc thời gian, lượng nước,
   tần suất, điều kiện hoặc lưu ý kỹ thuật thì
   ưu tiên trình bày cụ thể.

9. Chỉ khi TẤT CẢ các đoạn tài liệu hoàn toàn không
   liên quan đến câu hỏi mới trả lời:
   "Tôi chưa tìm thấy thông tin phù hợp trong tài liệu hiện có."

Hãy trả lời trực tiếp câu hỏi của người dùng.
"""

    # --------------------------------------------------------
    # 4. Gọi Gemini
    # --------------------------------------------------------

    try:
        answer = generate_content(
            prompt
        )

    except Exception as error:
        error_text = str(error)

        if (
            "giới hạn lượt sử dụng" in error_text.lower()
            or "quota" in error_text.lower()
            or "429" in error_text
        ):
            message = (
                "⚠️ Dịch vụ AI đang tạm đạt giới hạn lượt sử dụng. "
                "Vui lòng thử lại sau vài phút."
            )
        else:
            message = (
                "⚠️ Dịch vụ AI tạm thời chưa phản hồi. "
                "Vui lòng thử lại sau."
            )

        return (
            message,
            build_sources(results)
        )

    # --------------------------------------------------------
    # 5. Nguồn
    # --------------------------------------------------------

    sources = build_sources(
        results
    )

    return answer, sources


def analyze_crop_image(
    image,
    question: str = ""
) -> str:
    if image is None:
        raise ValueError(
            "Bạn chưa chọn ảnh."
        )

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")

    user_question = (
        question.strip()
        if question and question.strip()
        else "Hãy phân tích tình trạng cây trồng."
    )

    prompt = f"""
Bạn là chuyên gia hỗ trợ nông nghiệp tại Đắk Lắk.

Hãy phân tích ảnh theo cấu trúc:

LOẠI CÂY:
...

TRIỆU CHỨNG QUAN SÁT:
...

KHẢ NĂNG NGUYÊN NHÂN:
...

TRUY VẤN TÀI LIỆU:
...

Yêu cầu:
- Không kết luận chắc chắn bệnh chỉ từ một ảnh.
- Chỉ mô tả những dấu hiệu thực sự quan sát được.
- Nêu tối đa ba khả năng nguyên nhân.

Câu hỏi người dùng:
{user_question}
"""

    return generate_content(
        [prompt, image]
    )


def ask_image_rag(
    image,
    question: str = "",
    top_k: int = 3
):
    if image is None:
        return "Bạn chưa chọn ảnh.", "", ""

    analysis = analyze_crop_image(
        image,
        question
    )

    retrieval_query = (
        f"{question}\n{analysis}"
        if question and question.strip()
        else analysis
    )

    results = search(
        retrieval_query,
        k=int(top_k)
    )

    if not results:
        return (
            analysis,
            "Không tìm thấy tài liệu phù hợp.",
            ""
        )

    context = build_context(results)

    prompt = f"""
Bạn là ANGriTECH, trợ lý AI nông nghiệp Đắk Lắk.

PHÂN TÍCH HÌNH ẢNH:
{analysis}

CÂU HỎI:
{question or "Cây trồng trong ảnh có dấu hiệu gì?"}

TÀI LIỆU TRUY XUẤT:
{context}

Yêu cầu:
- Chỉ đưa khuyến nghị dựa trên tài liệu.
- Phân biệt dấu hiệu nhìn thấy từ ảnh và
  nội dung lấy từ tài liệu.
- Không khẳng định chắc chắn bệnh nếu chưa đủ bằng chứng.
- Không tự đề xuất thuốc hoặc liều lượng ngoài tài liệu.
- Kết thúc bằng câu:
  "Kết quả phân tích ảnh chỉ mang tính hỗ trợ."
"""

    answer = generate_content(prompt)

    return (
        analysis,
        answer,
        build_sources(results)
    )


# ============================================================
# 8. NHẬN DẠNG GIỌNG NÓI
# ============================================================

def transcribe_audio(
    audio_path,
    language_name: str
) -> str:
    if not audio_path:
        raise ValueError(
            "Bạn chưa ghi âm hoặc tải file."
        )

    mime_type, _ = mimetypes.guess_type(
        audio_path
    )

    mime_type = mime_type or "audio/wav"

    with open(audio_path, "rb") as file:
        audio_bytes = file.read()

    prompt = f"""
Hãy chép lại chính xác nội dung đoạn âm thanh.

Ngôn ngữ dự kiến:
{language_name}

Chỉ trả về phần văn bản được nhận dạng.
Không giải thích thêm.
"""

    return generate_content(
        [
            prompt,
            types.Part.from_bytes(
                data=audio_bytes,
                mime_type=mime_type
            )
        ]
    )


def voice_rag(
    audio_path,
    language: str
):
    try:
        question = transcribe_audio(
            audio_path,
            language
        )

        answer, sources = ask_rag_with_source(
            question
        )

        return question, answer, sources

    except Exception as error:
        return (
            "",
            f"Lỗi: {type(error).__name__}: {error}",
            ""
        )


# ============================================================
# 9. TIẾNG Ê ĐÊ THỬ NGHIỆM
# ============================================================

def ede_voice_rag(
    audio_path,
    answer_language: str
):
    try:
        ede_text = transcribe_audio(
            audio_path,
            "Tiếng Ê Đê tại Việt Nam"
        )

        translation_prompt = f"""
Bạn là trợ lý ngôn ngữ Ê Đê - Việt
trong lĩnh vực nông nghiệp.

Hãy chuyển câu tiếng Ê Đê sau thành
một câu truy vấn tiếng Việt ngắn, rõ nghĩa.

Không tự thêm thông tin.

CÂU TIẾNG Ê ĐÊ:
{ede_text}

Chỉ trả về câu truy vấn tiếng Việt.
"""

        vietnamese_query = generate_content(
            translation_prompt
        )

        answer, sources = ask_rag_with_source(
            vietnamese_query
        )

        if answer_language == "Ê Đê - Việt":
            bilingual_prompt = f"""
Hãy trình bày nội dung sau theo hai phần:

1. Tiếng Việt.
2. Tiếng Ê Đê đơn giản.

Nếu không chắc thuật ngữ chuyên môn tiếng Ê Đê,
hãy giữ nguyên thuật ngữ tiếng Việt.

NỘI DUNG:
{answer}
"""

            answer = generate_content(
                bilingual_prompt
            )

        recognized = (
            f"VĂN BẢN Ê ĐÊ NHẬN DẠNG:\n"
            f"{ede_text}\n\n"
            f"TRUY VẤN TIẾNG VIỆT:\n"
            f"{vietnamese_query}"
        )

        return recognized, answer, sources

    except Exception as error:
        return (
            "",
            f"Lỗi: {type(error).__name__}: {error}",
            ""
        )


# ============================================================
# 10. HÀM GIAO DIỆN
# ============================================================

def text_interface(question):
    return ask_rag_with_source(question)


def image_interface(
    image,
    question,
    top_k
):
    try:
        return ask_image_rag(
            image,
            question or "",
            int(top_k)
        )

    except Exception as error:
        return (
            "Không thể phân tích ảnh.",
            f"Lỗi: {type(error).__name__}: {error}",
            ""
        )


# ============================================================
# 11. CSS
# ============================================================


custom_css = """
@import url(
    'https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@400;500;600;700;800&display=swap'
);

* {
    font-family: 'Be Vietnam Pro', sans-serif !important;
    box-sizing: border-box;
}


/* ============================================================
   TOÀN TRANG
============================================================ */

body {
    background:
        linear-gradient(
            135deg,
            #f8fafc 0%,
            #eefbf3 50%,
            #f8fafc 100%
        ) !important;
}

.gradio-container {
    max-width: 1450px !important;
    margin: auto !important;
    padding: 20px !important;

    background: transparent !important;
}


/* ============================================================
   HEADER
============================================================ */

#header {
    position: relative;
    overflow: hidden;

    padding: 30px 24px !important;
    margin-bottom: 18px !important;

    border-radius: 26px !important;

    background:
        linear-gradient(
            135deg,
            #075f3d 0%,
            #0d7a4f 45%,
            #21935f 100%
        ) !important;

    border:
        1px solid rgba(255,255,255,.24) !important;

    box-shadow:
        0 18px 45px rgba(0,0,0,.18),
        inset 0 1px 0 rgba(255,255,255,.18) !important;
}

#header::before {
    content: "";

    position: absolute;
    inset: 0;

    background:
        linear-gradient(
            115deg,
            transparent 25%,
            rgba(255,255,255,.18) 50%,
            transparent 75%
        );

    transform: translateX(-130%);
    animation: headerShine 7s infinite;
}

@keyframes headerShine {
    0% {
        transform: translateX(-130%);
    }

    45%,
    100% {
        transform: translateX(130%);
    }
}

#header h1 {
    position: relative;
    z-index: 1;

    margin: 0 !important;

    color: #ffffff !important;

    font-size: clamp(38px, 5vw, 58px) !important;
    font-weight: 800 !important;

    letter-spacing: 1px !important;

    text-shadow:
        0 3px 0 rgba(0,0,0,.18),
        0 8px 22px rgba(0,0,0,.28),
        0 0 24px rgba(134,239,172,.30) !important;
}

#header p {
    position: relative;
    z-index: 1;

    margin-top: 10px !important;

    color: #ecfdf5 !important;

    font-size: 15px !important;
    font-weight: 600 !important;

    text-shadow:
        0 2px 8px rgba(0,0,0,.35) !important;
}


/* ============================================================
   THANH TAB
============================================================ */

.tab-nav {
    display: flex !important;
    gap: 8px !important;

    padding: 8px !important;
    margin-bottom: 16px !important;

    border-radius: 18px !important;

    background:
        linear-gradient(
            135deg,
            #064e3b 0%,
            #0b6848 100%
        ) !important;

    border:
        1px solid rgba(255,255,255,.14) !important;

    box-shadow:
        0 12px 30px rgba(0,0,0,.18),
        inset 0 1px 0 rgba(255,255,255,.12) !important;
}


/* ============================================================
   TAB CHƯA CHỌN
============================================================ */

.tab-nav button,
button[role="tab"] {
    position: relative !important;

    flex: 1 1 auto !important;

    min-height: 46px !important;

    padding: 11px 14px !important;

    color: #ffffff !important;

    font-size: 15px !important;
    font-weight: 800 !important;

    background:
        rgba(255,255,255,.08) !important;

    border:
        1px solid rgba(255,255,255,.10) !important;

    border-radius: 12px !important;

    opacity: 1 !important;

    text-shadow:
        0 2px 7px rgba(0,0,0,.58) !important;

    transition:
        transform .22s ease,
        background .22s ease,
        box-shadow .22s ease !important;
}


/* ÉP CHỮ VÀ ICON TRẮNG */

.tab-nav button *,
.tab-nav button span,
.tab-nav button div,
button[role="tab"] *,
button[role="tab"] span,
button[role="tab"] div {
    color: #ffffff !important;
    fill: #ffffff !important;
    opacity: 1 !important;
    font-weight: 800 !important;
}


/* HOVER */

.tab-nav button:hover,
button[role="tab"]:hover {
    color: #ffffff !important;

    background:
        linear-gradient(
            135deg,
            rgba(34,197,94,.40),
            rgba(16,185,129,.28)
        ) !important;

    transform: translateY(-2px);

    box-shadow:
        0 8px 18px rgba(22,163,74,.22) !important;
}


/* TAB ĐANG CHỌN */

.tab-nav button.selected,
.tab-nav button[aria-selected="true"],
button[role="tab"][aria-selected="true"] {
    color: #ffffff !important;

    background:
        linear-gradient(
            135deg,
            #f97316 0%,
            #fb923c 46%,
            #16a34a 100%
        ) !important;

    border:
        1px solid rgba(255,255,255,.30) !important;

    box-shadow:
        0 10px 24px rgba(249,115,22,.30),
        0 0 18px rgba(34,197,94,.16) !important;

    transform:
        translateY(-2px)
        scale(1.01);
}


/* ============================================================
   CARD
============================================================ */

.card {
    padding: 18px !important;

    border-radius: 20px !important;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,.98),
            rgba(242,253,247,.97)
        ) !important;

    border:
        1px solid rgba(22,101,52,.16) !important;

    box-shadow:
        0 16px 38px rgba(0,0,0,.12),
        inset 0 1px 0 rgba(255,255,255,.95) !important;
}


/* ============================================================
   LABEL
============================================================ */

.card label,
.card label span,
.card .label-wrap,
.card .label-wrap span {
    color: #14532d !important;

    font-size: 14px !important;
    font-weight: 800 !important;

    opacity: 1 !important;

    text-shadow: none !important;
}


/* ============================================================
   TEXTBOX / INPUT
============================================================ */

.card textarea,
.card input,
.card select {
    color: #111827 !important;

    background: #ffffff !important;

    border:
        1.5px solid rgba(22,101,52,.22) !important;

    border-radius: 13px !important;

    box-shadow:
        inset 0 2px 6px rgba(0,0,0,.035) !important;
}

.card textarea::placeholder,
.card input::placeholder {
    color: #6b7280 !important;
    opacity: 1 !important;
}

.card textarea:focus,
.card input:focus,
.card select:focus {
    border-color: #22c55e !important;

    box-shadow:
        0 0 0 4px rgba(34,197,94,.13) !important;
}


/* ============================================================
   OUTPUT
============================================================ */

.card textarea[readonly] {
    color: #111827 !important;

    background:
        linear-gradient(
            180deg,
            #ffffff 0%,
            #f8fffb 100%
        ) !important;
}


/* ============================================================
   NÚT CHÍNH
============================================================ */

button.primary {
    color: #ffffff !important;

    font-weight: 800 !important;

    background:
        linear-gradient(
            135deg,
            #f97316 0%,
            #fb7c18 45%,
            #16a34a 100%
        ) !important;

    border:
        1px solid rgba(255,255,255,.22) !important;

    border-radius: 13px !important;

    box-shadow:
        0 8px 18px rgba(249,115,22,.24),
        inset 0 1px 0 rgba(255,255,255,.18) !important;

    transition:
        transform .20s ease,
        box-shadow .20s ease,
        filter .20s ease !important;
}

button.primary:hover {
    transform: translateY(-2px);

    filter: brightness(1.05);

    box-shadow:
        0 11px 24px rgba(249,115,22,.30),
        0 0 18px rgba(34,197,94,.14) !important;
}


/* ============================================================
   NÚT PHỤ
============================================================ */

.card button:not(.primary) {
    color: #14532d !important;

    background:
        #ecfdf5 !important;

    border:
        1px solid rgba(22,101,52,.22) !important;

    font-weight: 700 !important;
}


/* ============================================================
   INFO / MARKDOWN
============================================================ */

.card p,
.card li,
.card strong,
.card em {
    color: #1f2937 !important;
}

.card h1,
.card h2,
.card h3,
.card h4 {
    color: #14532d !important;
}


/* ============================================================
   CẢNH BÁO
============================================================ */

.warning-box {
    padding: 12px 14px !important;

    border-radius: 12px !important;

    color: #92400e !important;

    background:
        #fff7ed !important;

    border:
        1px solid #fed7aa !important;
}


/* ============================================================
   RESPONSIVE MOBILE
============================================================ */

@media (max-width: 768px) {

    .gradio-container {
        padding: 10px !important;
    }

    #header {
        padding: 22px 14px !important;
        border-radius: 18px !important;
    }

    #header h1 {
        font-size: 38px !important;
    }

    #header p {
        font-size: 13px !important;
    }

    .tab-nav {
        flex-wrap: wrap !important;
    }

    .tab-nav button,
    button[role="tab"] {
        flex: 1 1 46% !important;

        font-size: 13px !important;

        min-height: 42px !important;

        padding: 9px 8px !important;
    }

    .card {
        padding: 12px !important;
        border-radius: 16px !important;
    }
}

footer {
    display: none !important;
}


/* ============================================================
   CELL 41 - FIX TAB CONTRAST
============================================================ */

/* Thanh chứa tab */
.tab-nav,
.tabs > .tab-nav,
div[role="tablist"] {
    gap: 6px !important;
    padding: 6px !important;

    background: #f0fdf4 !important;

    border: 1px solid #bbf7d0 !important;
    border-radius: 14px !important;

    box-shadow:
        0 5px 15px rgba(22, 101, 52, 0.08) !important;
}


/* ============================================================
   TAB CHƯA CHỌN
============================================================ */

.tab-nav button,
.tabs .tab-nav button,
button[role="tab"] {
    min-height: 44px !important;

    padding: 10px 14px !important;

    color: #14532d !important;

    background:
        linear-gradient(
            180deg,
            #ffffff 0%,
            #ecfdf5 100%
        ) !important;

    border: 1px solid #bbf7d0 !important;
    border-radius: 10px !important;

    font-size: 14px !important;
    font-weight: 800 !important;

    opacity: 1 !important;

    text-shadow: none !important;

    box-shadow:
        0 2px 6px rgba(0,0,0,.05) !important;

    transition:
        background .18s ease,
        color .18s ease,
        transform .18s ease,
        box-shadow .18s ease !important;
}


/* Ép text/span bên trong tab chưa chọn */
.tab-nav button span,
.tab-nav button div,
.tabs .tab-nav button span,
.tabs .tab-nav button div,
button[role="tab"] span,
button[role="tab"] div {
    color: #14532d !important;

    opacity: 1 !important;

    font-weight: 800 !important;

    text-shadow: none !important;
}


/* ============================================================
   HOVER
============================================================ */

.tab-nav button:hover,
.tabs .tab-nav button:hover,
button[role="tab"]:hover {
    color: #065f46 !important;

    background:
        linear-gradient(
            135deg,
            #dcfce7 0%,
            #d1fae5 100%
        ) !important;

    border-color: #4ade80 !important;

    transform: translateY(-1px) !important;

    box-shadow:
        0 6px 14px rgba(22,163,74,.14) !important;
}

.tab-nav button:hover span,
.tab-nav button:hover div,
button[role="tab"]:hover span,
button[role="tab"]:hover div {
    color: #065f46 !important;
}


/* ============================================================
   TAB ĐANG ĐƯỢC CHỌN
============================================================ */

.tab-nav button.selected,
.tab-nav button[aria-selected="true"],
.tabs .tab-nav button.selected,
.tabs .tab-nav button[aria-selected="true"],
button[role="tab"][aria-selected="true"] {
    color: #ffffff !important;

    background:
        linear-gradient(
            100deg,
            #ff7417 0%,
            #f97316 45%,
            #16a34a 100%
        ) !important;

    border-color: transparent !important;

    box-shadow:
        0 7px 17px rgba(249,115,22,.22),
        0 3px 10px rgba(22,163,74,.15) !important;

    transform: translateY(-1px) !important;

    text-shadow:
        0 1px 3px rgba(0,0,0,.30) !important;
}


/* Ép chữ tab đang chọn thành trắng */
.tab-nav button.selected span,
.tab-nav button.selected div,
.tab-nav button[aria-selected="true"] span,
.tab-nav button[aria-selected="true"] div,
.tabs .tab-nav button[aria-selected="true"] span,
.tabs .tab-nav button[aria-selected="true"] div,
button[role="tab"][aria-selected="true"] span,
button[role="tab"][aria-selected="true"] div {
    color: #ffffff !important;

    opacity: 1 !important;

    font-weight: 800 !important;

    text-shadow:
        0 1px 3px rgba(0,0,0,.30) !important;
}


/* ============================================================
   MOBILE
============================================================ */

@media (max-width: 768px) {

    .tab-nav,
    .tabs > .tab-nav,
    div[role="tablist"] {
        display: grid !important;

        grid-template-columns:
            repeat(2, minmax(0, 1fr)) !important;

        gap: 6px !important;
    }

    .tab-nav button,
    .tabs .tab-nav button,
    button[role="tab"] {
        width: 100% !important;

        min-height: 46px !important;

        padding: 8px 7px !important;

        font-size: 12px !important;

        white-space: normal !important;

        line-height: 1.25 !important;
    }
}


"""



# ============================================================
# 12. GIAO DIỆN GRADIO
# ============================================================

with gr.Blocks(
    title="ANGriTECH",
    css=custom_css
) as demo:

    gr.HTML(
        """
<div id="header">
    <h1>🌱 ANGriTECH</h1>
    <p>
        Trợ lý AI bản địa hỗ trợ sản xuất
        nông nghiệp Đắk Lắk
    </p>
</div>
"""
    )

    with gr.Tabs():

        with gr.Tab("📝 Tra cứu văn bản"):

            with gr.Group(
                elem_classes="card"
            ):
                text_question = gr.Textbox(
                    label="Nhập câu hỏi",
                    lines=4,
                    placeholder=(
                        "Ví dụ: Khi nào nên tưới nước "
                        "cho cây cà phê?"
                    )
                )

                text_button = gr.Button(
                    "🔍 Tra cứu",
                    variant="primary"
                )

                text_answer = gr.Textbox(
                    label="Câu trả lời",
                    lines=12
                )

                text_source = gr.Textbox(
                    label="Nguồn tài liệu",
                    lines=6
                )

                text_button.click(
                    fn=text_interface,
                    inputs=text_question,
                    outputs=[
                        text_answer,
                        text_source
                    ]
                )

        with gr.Tab("📷 Phân tích hình ảnh"):

            with gr.Row():

                with gr.Column(
                    elem_classes="card"
                ):
                    image_input = gr.Image(
                        type="pil",
                        label="Ảnh cây trồng"
                    )

                    image_question = gr.Textbox(
                        label="Câu hỏi",
                        lines=3
                    )

                    top_k_input = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Số đoạn tài liệu"
                    )

                    image_button = gr.Button(
                        "🔍 Phân tích",
                        variant="primary"
                    )

                with gr.Column(
                    elem_classes="card"
                ):
                    image_analysis = gr.Textbox(
                        label="Phân tích hình ảnh",
                        lines=9
                    )

                    image_answer = gr.Textbox(
                        label="Tư vấn từ tài liệu",
                        lines=12
                    )

                    image_source = gr.Textbox(
                        label="Nguồn",
                        lines=6
                    )

            image_button.click(
                fn=image_interface,
                inputs=[
                    image_input,
                    image_question,
                    top_k_input
                ],
                outputs=[
                    image_analysis,
                    image_answer,
                    image_source
                ]
            )

        with gr.Tab("🎙️ Giọng nói Việt / Anh"):

            with gr.Group(
                elem_classes="card"
            ):
                voice_language = gr.Dropdown(
                    choices=[
                        "Tiếng Việt",
                        "English"
                    ],
                    value="Tiếng Việt",
                    label="Ngôn ngữ"
                )

                voice_input = gr.Audio(
                    sources=[
                        "microphone",
                        "upload"
                    ],
                    type="filepath",
                    label="Ghi âm hoặc tải file"
                )

                voice_button = gr.Button(
                    "🎙️ Nhận dạng và tra cứu",
                    variant="primary"
                )

                voice_text = gr.Textbox(
                    label="Nội dung nhận dạng",
                    lines=4
                )

                voice_answer = gr.Textbox(
                    label="Câu trả lời",
                    lines=12
                )

                voice_source = gr.Textbox(
                    label="Nguồn",
                    lines=6
                )

                voice_button.click(
                    fn=voice_rag,
                    inputs=[
                        voice_input,
                        voice_language
                    ],
                    outputs=[
                        voice_text,
                        voice_answer,
                        voice_source
                    ]
                )

        with gr.Tab("🗣️ Tiếng Ê Đê - thử nghiệm"):

            with gr.Group(
                elem_classes="card"
            ):
                gr.Markdown(
                    """
Chức năng tiếng Ê Đê đang trong giai đoạn thử nghiệm.
Người dùng nên kiểm tra nội dung nhận dạng.
"""
                )

                ede_audio = gr.Audio(
                    sources=[
                        "microphone",
                        "upload"
                    ],
                    type="filepath",
                    label="Âm thanh tiếng Ê Đê"
                )

                ede_language = gr.Dropdown(
                    choices=[
                        "Tiếng Việt",
                        "Ê Đê - Việt"
                    ],
                    value="Tiếng Việt",
                    label="Ngôn ngữ trả lời"
                )

                ede_button = gr.Button(
                    "🔍 Nhận dạng và tra cứu",
                    variant="primary"
                )

                ede_text = gr.Textbox(
                    label="Nội dung nhận dạng",
                    lines=8
                )

                ede_answer = gr.Textbox(
                    label="Câu trả lời",
                    lines=12
                )

                ede_source = gr.Textbox(
                    label="Nguồn",
                    lines=6
                )

                ede_button.click(
                    fn=ede_voice_rag,
                    inputs=[
                        ede_audio,
                        ede_language
                    ],
                    outputs=[
                        ede_text,
                        ede_answer,
                        ede_source
                    ]
                )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
