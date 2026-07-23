from __future__ import annotations
import base64, io

class OCRService:
    def __init__(self, dify=None):
        self.dify = dify

    def extract(self, image_base64: str, correction: str = "") -> dict:
        if correction.strip():
            return {"text": correction.strip(), "engine": "学生校正文本", "confidence": 1.0, "needs_correction": False}
        if self.dify and self.dify.enabled and image_base64:
            text = self.dify.vision_ocr(image_base64)
            if text:
                return {"text": text, "engine": "Dify 多模态 OCR", "confidence": 0.9, "needs_correction": False}
        try:
            from PIL import Image, ImageOps, ImageEnhance
            import pytesseract
            raw = image_base64.split(",",1)[-1]
            image = Image.open(io.BytesIO(base64.b64decode(raw)))
            image = ImageOps.exif_transpose(image).convert("L")
            image = ImageEnhance.Contrast(image).enhance(1.8)
            if max(image.size)>2200:
                image.thumbnail((2200,2200))
            text = pytesseract.image_to_string(image, lang="chi_sim+eng").strip()
            return {"text": text, "engine": "本地 Tesseract OCR", "confidence": 0.75 if text else 0.0, "needs_correction": not bool(text)}
        except Exception:
            return {"text": "", "engine": "OCR 待配置", "confidence": 0.0, "needs_correction": True,
                    "message": "当前未安装中文 OCR。请在下方校正框输入图片题干，或配置 Dify 多模态模型。"}
