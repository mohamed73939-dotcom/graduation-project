from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
from pathlib import Path
from logger_config import summarizer_logger

AR_STOPWORDS = {
    "في","من","على","إلى","الى","عن","أن","إن","او","أو","و","ثم","لكن","بل","كما","مع","كان","كانت",
    "هذه","هذا","ذلك","تلك","هو","هي","هم","هن","انا","أنا","انت","أنت","انتم","إنه","إنها","هناك",
    "قد","قد","ما","لا","لم","لن","حتى","كل","أي","أى","أو","إذا","إذ","كما","حيث","مثل","علشان","لان",
    "بس","ده","دي","ده","وده","ودي","مش","ليس","كانت","كانوا","تم","قد","قد","جدا","جدًا","كذا","كذاك"
}
EN_STOPWORDS = {
    "the","a","an","and","or","but","so","to","of","in","on","at","by","for","with","as","is","are","was","were",
    "it","its","this","that","these","those","he","she","they","we","you","i","be","been","being","do","does","did",
    "not","no","yes","from","into","than","then","there","here","very","also","just","only","about","over","under",
    "their","his","her","our","your","them"
}

def _normalize_ar_en(text: str) -> str:
    # remove diacritics, tatweel, punctuation; keep arabic/latin letters and digits
    text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)  # Arabic diacritics + tatweel
    text = re.sub(r'[^\w\u0600-\u06FF\s]', ' ', text, flags=re.UNICODE)  # strip punctuation but keep Arabic
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def _tokens(text: str):
    t = _normalize_ar_en(text)
    raw = t.split()
    # simple stopword filter
    filtered = []
    for w in raw:
        if w in AR_STOPWORDS or w in EN_STOPWORDS:
            continue
        filtered.append(w)
    return set(filtered)

class LectureSummarizer:
    """
    Grounded hierarchical summarization with automatic re-roll and extractive fallback.
    """
    def __init__(self, model_name="csebuetnlp/mT5_multilingual_XLSum", output_dir="outputs/summaries", strategy="hybrid"):
        self.device = 0 if torch.cuda.is_available() else -1
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.strategy = strategy
        
        # Attribute holders
        self.student_pipeline = None
        self.student_tokenizer = None
        self.teacher_pipeline = None
        self.teacher_tokenizer = None
        
        # Load Primary Model (Student)
        if self.strategy == "hybrid":
             # In hybrid mode, model_name passed in is treated as Student unless fallback logic differs
             # Ideally we check configured paths. For now, assume model_name is the "Student"/Primary.
             self._load_student_model(model_name)
        else:
             # Single model mode
             self._load_student_model(model_name)


    def _load_student_model(self, model_name):
        summarizer_logger.info(f"Loading Student model: {model_name}")
        try:
            self.student_tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.student_pipeline = pipeline("summarization", model=model, tokenizer=self.student_tokenizer, device=self.device)
            summarizer_logger.info("✓ Student model loaded")
        except Exception as e:
            summarizer_logger.error(f"Student model load failed: {e}")
            # Fallback to teacher immediately if student fails?
            self._load_teacher_model()
            self.student_pipeline = self.teacher_pipeline # Alias it
            self.student_tokenizer = self.teacher_tokenizer

    def _load_teacher_model(self):
        if self.teacher_pipeline is not None:
            return # Already loaded
            
        TEACHER_MODEL = "csebuetnlp/mT5_multilingual_XLSum"
        summarizer_logger.info(f"Loading Teacher model (Lazy Load): {TEACHER_MODEL}")
        try:
            self.teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
            model = AutoModelForSeq2SeqLM.from_pretrained(TEACHER_MODEL)
            self.teacher_pipeline = pipeline("summarization", model=model, tokenizer=self.teacher_tokenizer, device=self.device)
            summarizer_logger.info("✓ Teacher model loaded")
        except Exception as e:
            summarizer_logger.error(f"Teacher model load failed! {e}")
            # Extreme fallback
            self.teacher_pipeline = self.student_pipeline 


    # Public API ---------------------------------------------------------------

    def summarize(self, text, max_length=450, min_length=80, language='ar', **kwargs):
        """Wrapper for hierarchical_summarize to match expected API"""
        res, metadata = self.hierarchical_summarize(
            text,
            language=language,
            final_max_length=max_length,
            min_length=min_length,
            **kwargs
        )
        return res

    def hierarchical_summarize(
        self,
        text,
        language='ar',
        intermediate_max_length=220,
        final_max_length=450,
        min_length=80,
        chunk_token_limit=900,
        beam_size=4,
        constraints=None
    ):
        clean = self._preprocess_text(text, language)
        if len(clean.split()) < 50:
            msg = "النص قصير جداً للتلخيص" if language == 'ar' else "Text too short to summarize."
            return msg, {"short": True, "word_count": len(clean.split())}

        chunks = self._tokenizer_aware_chunks(clean, max_tokens=chunk_token_limit)
        summarizer_logger.info(f"Hierarchical: {len(chunks)} base chunks")

        # Intermediate summaries - Batch Processing
        intermediate_prompts = [self._build_intermediate_prompt(c, language) for c in chunks]
        
        # Run batch inference
        # Use beam_size=1 (greedy) for speed on intermediate steps
        intermediate_results = self._transcribe_batch(
            intermediate_prompts,
            max_length=intermediate_max_length,
            min_length=min_length,
            num_beams=1  # Fast greedy search for intermediates
        )
        intermediate_summaries = [self._clean_generated_text(r) for r in intermediate_results]

        combined_text = "\n".join(intermediate_summaries)
        summarizer_logger.info(f"Intermediate combined words={len(combined_text.split())}")

        # Final prompt (grounded)
        final_prompt = self._build_final_prompt(
            combined_text,
            language=language,
            constraints=constraints
        )
        
        # Single final summary with slightly better quality (beam_size=2)
        # Still lower than original 4 for speed, but better than 1
        final_beams = max(2, beam_size // 2)
        
        final_summary_res = self._run_model(
            final_prompt,
            max_length=final_max_length,
            min_length=min_length,
            num_beams=final_beams,
            pipeline_instance=self.student_pipeline
        )
        final_summary = self._post_process(final_summary_res, language)

        # Grounding check
        check = self.ground_check(final_summary, clean, language)
        rerolled = False
        fallback_used = False
        
        # HYBRID STRATEGY CHECK
        # If score is low and we are in hybrid mode, try Teacher Model BEFORE resorting to re-rolling or extractive
        if not check["ok"] and self.strategy == "hybrid":
             summarizer_logger.warning(f"Student weak (score={check['score']:.2f}). Hybrid Switch -> Loading Teacher.")
             self._load_teacher_model()
             if self.teacher_pipeline:
                  # Run Teacher
                  teacher_res = self._run_model(
                      final_prompt,
                      max_length=final_max_length,
                      min_length=min_length,
                      num_beams=final_beams, 
                      pipeline_instance=self.teacher_pipeline
                  )
                  teacher_summary = self._post_process(teacher_res, language)
                  check_teacher = self.ground_check(teacher_summary, clean, language)
                  
                  # If teacher is better, use it
                  if check_teacher["score"] > check["score"]:
                       summarizer_logger.info(f"Teacher improvement: {check['score']:.2f} -> {check_teacher['score']:.2f}")
                       final_summary = teacher_summary
                       check = check_teacher
                       metadata_model_used = "teacher_xlsum"  # We'll need to inject this into metadata later
                  else:
                       summarizer_logger.info("Teacher result was not better. Keeping Student result.")

        if not check["ok"]:
            summarizer_logger.warning(f"Grounding weak (score={check['score']:.2f}, offenders={check['bad_tokens'][:5]}). Re-rolling with stricter prompt.")
            strict_prompt = self._build_final_prompt(
                combined_text,
                language=language,
                constraints={
                    **(constraints or {}),
                    "no_external_entities": True,
                    "format": "bullets+paragraphs",
                    "strict": True
                }
            )
            # Re-roll using the *Current Best* pipeline (which might be teacher now if we switched)
            active_pipe = self.teacher_pipeline if (self.teacher_pipeline and self.strategy=="hybrid") else self.student_pipeline
            
            final_summary2 = self._run_model(
                strict_prompt,
                max_length=max(280, int(final_max_length * 0.75)),  # shorter to reduce drift
                min_length=min(120, max(60, int(min_length * 0.75))),
                num_beams=max(5, beam_size),  # slightly larger beam
                pipeline_instance=active_pipe
            )
            final_summary2 = self._post_process(final_summary2, language)
            check2 = self.ground_check(final_summary2, clean, language)
            rerolled = True

            if check2["ok"] or check2["score"] >= max(0.55, check["score"]):
                final_summary = final_summary2
                check = check2
            else:
                summarizer_logger.warning("Grounding still weak after re-roll. Falling back to extractive summary.")
                fallback_used = True
                # Basic extractive fallback from the original text
                sentences = self._simple_sentence_split(clean)
                final_summary = self.extractive_fallback(sentences, language=language)

        # Hybrid Fusion: Prepend "Hard" Extractive Sentences
        extractive_basis = constraints.get("extractive_summary", "") if constraints else ""
        if extractive_basis and not fallback_used: # Don't duplicate if we already fell back to extractive
            # Take top 10 sentences from basis (increased per user request)
            top_facts = [s.strip() for s in extractive_basis.split('\n') if s.strip()][:10]
            if top_facts:
                if language == 'ar':
                    fact_header = f"📌 أهم {len(top_facts)} جمل وردت في الفيديو (نصياً):"
                    facts = "\n".join([f"- {f}" for f in top_facts])
                    final_summary = f"{fact_header}\n{facts}\n\n" + final_summary
                else:
                    fact_header = f"📌 Top {len(top_facts)} Verbatim Quotes:"
                    facts = "\n".join([f"- {f}" for f in top_facts])
                    final_summary = f"{fact_header}\n{facts}\n\n" + final_summary

        metadata = {
            "short": False,
            "base_chunk_count": len(chunks),
            "intermediate_count": len(intermediate_summaries),
            "model_used": self.model_name,
            "final_word_count": len(final_summary.split()),
            "grounded_score": round(check["score"], 3),
            "rerolled": rerolled,
            "fallback": fallback_used
        }
        return final_summary, metadata

    # Grounding utilities ------------------------------------------------------

    def is_grounded(self, summary: str, source_text: str) -> bool:
        return self.ground_check(summary, source_text)["ok"]

    def grounded_score(self, summary: str, source_text: str) -> float:
        return self.ground_check(summary, source_text)["score"]

    def ground_check(self, summary: str, source_text: str, language: str = 'ar'):
        # token coverage score: proportion of summary tokens present in source
        s_tokens = _tokens(summary)
        src_tokens = _tokens(source_text)
        if not s_tokens:
            return {"ok": False, "score": 0.0, "bad_tokens": []}
            
        # Detect T5 Malformed Artifacts (Sentinel tokens)
        # If the summary contains "extra_id", the model failed (hallucinated training tokens)
        if "extra_id" in summary or "<extra_id" in summary:
             summarizer_logger.warning("Detected T5 sentinel tokens (malformed output). Forcing score 0.0.")
             return {"ok": False, "score": 0.0, "bad_tokens": ["MALFORMED_OUTPUT"]}

        overlap = s_tokens & src_tokens
        score = len(overlap) / max(1, len(s_tokens))

        # banned hallucination hints (Arabic social/media generic phrases)
        banned_ar = {"فيسبوك","فيس بوك","تويتر","سلسلة من القضايا","موقع التواصل","موقع تويتر","شبكات التواصل"}
        banned_en = {"facebook","twitter","bbc","series","weekly"}
        bad = [w for w in s_tokens if (w in banned_ar or w in banned_en)]

        ok = score >= 0.35 and len(bad) == 0  # threshold tuned for short educational clips
        return {"ok": ok, "score": float(score), "bad_tokens": bad}

    def extractive_fallback(self, sentences, language='ar', limit=12):
        if not sentences:
            return "UNABLE_TO_SUMMARIZE: no sentences."
        selected = sentences[:limit]
        # Produce bullets + short paragraphs
        bullets = [("• " + s.strip()) for s in selected[:6]]
        paras = []
        block = []
        for s in selected[6:]:
            block.append(s.strip())
            if len(block) >= 3:
                paras.append(" ".join(block))
                block = []
        if block:
            paras.append(" ".join(block))
        header = "النقاط الرئيسية:\n\n" if language == 'ar' else "Key Points:\n\n"
        details = "التفاصيل:\n\n" if language == 'ar' else "Details:\n\n"
        return header + "\n".join(bullets) + ("\n\n" + details + "\n\n".join(paras) if paras else "")

    # Prompt builders ----------------------------------------------------------

    def _build_intermediate_prompt(self, text, language):
        if language == 'ar':
            return (
                "summarize: لخص بدقة وبأسلوب محايد اعتماداً فقط على النص التالي، "
                "بدون أي معلومات خارجية أو أمثلة غير موجودة:\n" + text
            )
        return "summarize: Provide a concise summary using ONLY information present:\n" + text

    def _build_final_prompt(self, text, language, constraints=None):
        constraints = constraints or {}
        grounding_text = constraints.get("grounding_text", "")
        keywords = constraints.get("keywords", [])
        no_external = constraints.get("no_external_entities", True)
        fmt = constraints.get("format", "bullets+paragraphs")
        strict = constraints.get("strict", False)

        keywords_str = ", ".join(keywords) if keywords else ""
        fmt_note = "ابدأ بـ 5-8 نقاط رئيسية، ثم 2-4 فقرات قصيرة." if language == 'ar' else "Start with 5-8 bullet points, then 2-4 short paragraphs."
        strict_note_ar = "لا تستخدم أي كلمة أو كيان غير مذكورين في النص. تجنب العموميات." if strict else ""
        strict_note_en = "Do not use any word/entity not present in the text. Avoid generic claims." if strict else ""

        if language == 'ar':
            prefix = (
                "summarize: لخص النص تعليميًا بشكل أمين دون إضافة حقائق خارجية.\n"
                "احصر التلخيص في موضوع القواعد المذكورة في النص (مثل present simple، s/es/ies، النفي بـ don't/doesn't، الأسئلة، الأمثلة الفعلية المذكورة).\n"
                f"{'الكلمات المفتاحية (اختياري): ' + keywords_str if keywords_str else ''}\n"
                f"التنسيق: {fmt_note}\n"
                f"{strict_note_ar}\n"
            )
        else:
            prefix = (
                "summarize: Produce a factual grounded summary. Use ONLY information from the transcript.\n"
                "Focus on the grammar points actually discussed (e.g., present simple, s/es/ies, don't/doesn't, questions, examples).\n"
                f"{'Keywords (optional): ' + keywords_str if keywords_str else ''}\n"
                f"Format: {fmt_note}\n"
                f"{strict_note_en}\n"
            )

        ocr_context = constraints.get("ocr_context", "")
        
        if ocr_context:
            # Inject visual text
            truncated_ocr = ocr_context[:3000] # Safety limit
            if language == 'ar':
                text = text + f"\n\n[نص مرئي من الشرائح]:\n{truncated_ocr}"
            else:
                text = text + f"\n\n[Visual Text from Slides]:\n{truncated_ocr}"
                
        if grounding_text:
            # Provide limited extra context (truncated) to improve grounding
            text = text + "\n\nContext (do not invent; for grounding only):\n" + grounding_text[:4000]
        return prefix + "\n" + text

    # Core model runner --------------------------------------------------------

    # Core model runner --------------------------------------------------------

    def _run_model(self, text, max_length, min_length, num_beams, pipeline_instance=None):
        # Default to student if not specified
        pipe = pipeline_instance if pipeline_instance else self.student_pipeline
        
        # Add mT5 style prefix if needed
        # We assume both models are T5-based for now.
        if not text.lower().startswith("summarize:"):
             text = "summarize: " + text

        try:
            result = pipe(
                text,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=2.0,
                no_repeat_ngram_size=4,
                do_sample=False
            )
            summary_text = result[0].get('summary_text', result[0].get('generated_text', '')).strip()
        except Exception as e:
            summarizer_logger.warning(f"Summarization error: {e}")
            summary_text = " "
        return self._clean_generated_text(summary_text)

    def _transcribe_batch(self, texts, max_length, min_length, num_beams):
        """Helper for batch processing"""
        if not texts:
            return []
        
        # Add prefix if needed
        if "mT5" in self.model_name or "mt5" in self.model_name.lower():
            texts = [("summarize: " + t if not t.lower().startswith("summarize:") else t) for t in texts]

        try:
            # Batch inference
            # pipeline handles list inputs by default
            results = self.summarizer(
                texts,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=2.0,
                no_repeat_ngram_size=4,
                do_sample=False,
                batch_size=4 
            )
            # pipeline returns list of dicts
            return [r.get('summary_text', r.get('generated_text', '')).strip() for r in results]
        except Exception as e:
            summarizer_logger.warning(f"Batch summarization error: {e}")
            return [" " for _ in texts]

    # Text utilities -----------------------------------------------------------

    def _preprocess_text(self, text, language):
        # Normalize separators and whitespace; keep digits and code tokens for technical content
        text = re.sub(r'(#+|\*{2,}|={2,}|-{3,})', ' ', text)
        # Collapse long laughter/elongation in Arabic (ههههه / وووو)
        text = re.sub(r'(ه){3,}', 'هه', text)
        text = re.sub(r'(و){3,}', 'و', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        if language == 'ar':
            text = re.sub(r'[\u064B-\u065F]', '', text)  # remove arabic diacritics
        return text.strip()

    def _simple_sentence_split(self, text):
        parts = re.split(r'[.!?؟؛\n]+', text)
        return [p.strip() for p in parts if p.strip()]

    def _tokenizer_aware_chunks(self, text, max_tokens=900):
        sentences = self._simple_sentence_split(text)
        chunks, current = [], ""
        # Default to student tokenizer
        tok = self.student_tokenizer if self.student_tokenizer else self.teacher_tokenizer
        
        for sent in sentences:
            try:
                sent_toks = len(tok.encode(sent, add_special_tokens=False))
                curr_toks = len(tok.encode(current, add_special_tokens=False)) if current else 0
            except Exception:
                sent_toks = len(sent.split())
                curr_toks = len(current.split()) if current else 0

            if curr_toks + sent_toks > max_tokens:
                if current:
                    chunks.append(current.strip())
                current = sent
            else:
                current = f"{current} {sent}".strip() if current else sent
        if current:
            chunks.append(current.strip())
        return chunks

    def _clean_generated_text(self, text):
        # Remove excessive repeats & clean spaces
        text = re.sub(r'(.)\1{6,}', r'\1\1\1', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _post_process(self, summary, language):
        summary = summary.strip()
        summary = re.sub(r'([.!?؟؛])\s*', r'\1 ', summary)
        # Guarantee ending punctuation
        if summary and summary[-1] not in '.؟!؛':
            summary += '.' if language != 'ar' else '؛'
        # Paragraph formatting: about 3-4 sentences each
        sentences = re.split(r'([.!?؟؛])', summary)
        paragraphs, current, count = [], [], 0
        for i in range(0, len(sentences), 2):
            sent = sentences[i].strip()
            punct = sentences[i+1] if i+1 < len(sentences) else ''
            if sent:
                current.append(sent + punct)
                count += 1
                if count >= 4:
                    paragraphs.append(" ".join(current))
                    current, count = [], 0
        if current:
            paragraphs.append(" ".join(current))

        formatted = "\n\n".join(paragraphs)

        # Add key points (first sentences)
        if len(paragraphs) >= 2:
            bullets = []
            for para in paragraphs[:6]:
                first_sentence = re.split(r'[.!?؟؛]', para)[0].strip()
                if len(first_sentence.split()) > 3:
                    bullets.append(f"• {first_sentence}")
            if bullets:
                header = "النقاط الرئيسية (من النموذج):\n\n" if language == 'ar' else "Key Points (AI Generated):\n\n"
                details = "التفاصيل:\n\n" if language == 'ar' else "Details:\n\n"
                formatted = header + "\n".join(bullets) + "\n\n" + details + formatted
        return formatted