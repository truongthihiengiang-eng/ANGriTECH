
import os
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
    last_error = None

    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents
            )

            if response.text:
                return response.text.strip()

        except Exception as error:
            last_error = error

            if attempt < retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(
        f"Không gọi được Gemini: {last_error}"
    )


# ============================================================
# 5. TÌM KIẾM FAISS
# ============================================================

def search(question: str, k: int = 5) -> list[dict]:
    if not question or not question.strip():
        return []

    query_vectors = list(
        embedding_model.query_embed(
            [question.strip()]
        )
    )

    query_vector = np.asarray(
        query_vectors,
        dtype=np.float32
    )

    faiss.normalize_L2(query_vector)

    scores, indices = index.search(
        query_vector,
        min(int(k), index.ntotal)
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):
        if idx < 0 or idx >= len(metadata):
            continue

        item = dict(metadata[idx])
        item["score"] = float(score)

        results.append(item)

    return results


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
    k: int = 5
):
    if not question or not question.strip():
        return "Bạn chưa nhập câu hỏi.", ""

    results = search(
        question,
        k=k
    )

    if not results:
        return (
            "Không tìm thấy tài liệu phù hợp.",
            ""
        )

    context = build_context(results)

    prompt = f"""
Bạn là ANGriTECH, trợ lý AI hỗ trợ sản xuất
nông nghiệp tại Đắk Lắk.

Chỉ sử dụng nội dung trong phần TÀI LIỆU
để trả lời câu hỏi.

Nếu tài liệu không đủ thông tin, phải nói rõ:
"Tôi không tìm thấy thông tin đầy đủ trong tài liệu."

TÀI LIỆU:
{context}

CÂU HỎI:
{question}

Yêu cầu:
- Trả lời bằng tiếng Việt.
- Trình bày rõ ràng và dễ hiểu.
- Không tự bổ sung thông tin ngoài tài liệu.
- Không tự đề xuất thuốc hoặc liều lượng
  nếu tài liệu không nêu.
"""

    answer = generate_content(prompt)

    return answer, build_sources(results)


# ============================================================
# 7. PHÂN TÍCH HÌNH ẢNH
# ============================================================

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
.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}

#header {
    padding: 28px;
    margin-bottom: 18px;
    border-radius: 24px;

    color: white;
    text-align: center;

    background:
        linear-gradient(
            135deg,
            rgba(3, 70, 40, 0.96),
            rgba(14, 130, 75, 0.92)
        );

    box-shadow:
        0 18px 45px rgba(0, 0, 0, 0.25);
}

#header h1 {
    margin: 0;
    font-size: 52px;
    font-weight: 800;
}

.tab-nav {
    padding: 8px !important;
    border-radius: 16px !important;

    background:
        linear-gradient(
            135deg,
            #064e3b,
            #166534
        ) !important;
}

.tab-nav button,
.tab-nav button *,
button[role="tab"],
button[role="tab"] * {
    color: white !important;
    opacity: 1 !important;
    font-weight: 800 !important;
}

.tab-nav button[aria-selected="true"] {
    background:
        linear-gradient(
            135deg,
            #ea580c,
            #f97316,
            #15803d
        ) !important;

    border-radius: 12px !important;
}

.card {
    padding: 18px !important;
    border-radius: 20px !important;

    background:
        rgba(248, 255, 250, 0.97) !important;

    border:
        1px solid rgba(22, 101, 52, 0.18) !important;

    box-shadow:
        0 16px 40px rgba(0, 0, 0, 0.14) !important;
}

.card label,
.card label span {
    color: #14532d !important;
    font-weight: 800 !important;
}

.card textarea,
.card input {
    color: #111827 !important;
    background: white !important;
}

footer {
    display: none !important;
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
