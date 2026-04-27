import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from pathlib import Path

# Optional UI dependency.
try:
    import streamlit as st  # type: ignore
    STREAMLIT_AVAILABLE = True
except ModuleNotFoundError:
    st = None
    STREAMLIT_AVAILABLE = False

# Optional model dependencies are loaded lazily so this file can still run
# in environments where Streamlit or TTS is unavailable.
try:
    import torch  # type: ignore
except ModuleNotFoundError:
    torch = None

MODEL_NAME = "tts_models/multilingual/multi-dataset/your_tts"
DEFAULT_LANGUAGE_CODE = "hi"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "generated_outputs"
PROFILE_DIR = BASE_DIR / "user_profiles"
TEMP_DIR = BASE_DIR / "temp_uploads"
OUTPUT_DIR.mkdir(exist_ok=True)
PROFILE_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)


@dataclass
class AppPaths:
    base_dir: Path
    output_dir: Path
    profile_dir: Path
    temp_dir: Path


APP_PATHS = AppPaths(BASE_DIR, OUTPUT_DIR, PROFILE_DIR, TEMP_DIR)


class UploadedFileAdapter:
    """Small adapter to support both Streamlit uploads and CLI file handling."""

    def __init__(self, path: Path):
        self.path = path
        self.name = path.name

    def read(self) -> bytes:
        return self.path.read_bytes()


class TTSEngine:
    """Lazy loader for YourTTS so helper utilities remain usable without TTS installed."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._tts = None

    def _gpu_available(self) -> bool:
        return bool(torch is not None and torch.cuda.is_available())

    def load(self):
        if self._tts is not None:
            return self._tts
        try:
            from TTS.api import TTS  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TTS is not installed in this environment. Install TTS in the active "
                "environment before generating speech."
            ) from exc
        self._tts = TTS(model_name=self.model_name, progress_bar=False, gpu=self._gpu_available())
        #self._tts = TTS(model_name=self.model_name, progress_bar=False, gpu=True)
        return self._tts

    def generate(self, text: str, speaker_wav: str, language_code: str, output_path: Path) -> Path:
        tts = self.load()
        tts.tts_to_file(
        text=" ".join(text.split()),
        speaker_wav=speaker_wav,
        language=language_code.strip(),
        file_path=str(output_path),
    )
        return output_path


ENGINE = TTSEngine()


def sanitize_name(value: str) -> str:
    cleaned = "".join(ch for ch in value if ch.isalnum() or ch in ("_", "-", " ")).strip()
    return cleaned.replace(" ", "_") or "user"



def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()



def profile_json_path(username: str, paths: AppPaths = APP_PATHS) -> Path:
    return paths.profile_dir / f"{sanitize_name(username)}.json"



def profile_audio_path(username: str, source_suffix: str = ".wav", paths: AppPaths = APP_PATHS) -> Path:
    return paths.profile_dir / f"{sanitize_name(username)}{source_suffix}"



def save_uploaded_file(uploaded_file: Any, username: str, paths: AppPaths = APP_PATHS) -> Path:
    suffix = Path(getattr(uploaded_file, "name", "sample.wav")).suffix or ".wav"
    target = paths.temp_dir / f"{sanitize_name(username)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"
    with open(target, "wb") as f:
        f.write(uploaded_file.read())
    return target



def preprocess_user_audio(source_path: Path, username: str, paths: AppPaths = APP_PATHS) -> Path:
    destination = profile_audio_path(username, source_path.suffix or ".wav", paths=paths)
    shutil.copy2(source_path, destination)
    return destination



def create_account(
    username: str,
    password: str,
    language_code: str,
    uploaded_voice: Any,
    paths: AppPaths = APP_PATHS,
) -> tuple[bool, str]:
    username = sanitize_name(username)
    profile_path = profile_json_path(username, paths=paths)

    if profile_path.exists():
        return False, "Username already exists. Please choose another username."

    raw_voice = save_uploaded_file(uploaded_voice, username, paths=paths)
    processed_audio = preprocess_user_audio(raw_voice, username, paths=paths)

    profile = {
        "username": username,
        "password_hash": hash_password(password),
        "language_code": language_code.strip() or DEFAULT_LANGUAGE_CODE,
        "voice_sample": str(processed_audio),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8")
    return True, "Account created successfully."



def authenticate_user(username: str, password: str, paths: AppPaths = APP_PATHS) -> tuple[bool, Optional[dict[str, Any]]]:
    profile_path = profile_json_path(username, paths=paths)
    if not profile_path.exists():
        return False, None

    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("password_hash") != hash_password(password):
        return False, None

    return True, profile



def build_output_path(output_prefix: str, paths: AppPaths = APP_PATHS) -> Path:
    return paths.output_dir / f"{sanitize_name(output_prefix)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"



def generate_speech(text: str, speaker_wav: str, language_code: str, output_prefix: str, paths: AppPaths = APP_PATHS) -> Path:
    output_path = build_output_path(output_prefix, paths=paths)
    return ENGINE.generate(text=text, speaker_wav=speaker_wav, language_code=language_code, output_path=output_path)



def run_streamlit_app() -> None:
    if not STREAMLIT_AVAILABLE or st is None:
        raise RuntimeError("Streamlit is not installed.")

    st.set_page_config(
        page_title="VoiceSphere AI",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.current_user = None
        st.session_state.profile = None
        st.session_state.page = "login"

    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 20% 30%, rgba(0, 255, 255, 0.10), transparent 30%),
                radial-gradient(circle at 80% 70%, rgba(0, 180, 200, 0.12), transparent 35%),
                linear-gradient(135deg, #020b10 0%, #061a20 50%, #031015 100%);
            color: white;
        }

        header, footer {
            visibility: hidden;
        }

        .main-container {
            min-height: 88vh;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 5rem;
            padding: 2rem 5rem;
        }

        .left-section {
            width: 48%;
        }

        .left-section h1 {
            font-size: 4rem;
            line-height: 1.1;
            color: #d9ffff;
            text-shadow: 0 0 18px rgba(0,255,255,0.45);
            margin-bottom: 1.5rem;
            font-weight: 800;
        }

        .left-section p {
            font-size: 1.35rem;
            color: #e3f7f8;
            line-height: 1.4;
            margin-bottom: 2rem;
        }

        .features {
            font-size: 1.25rem;
            line-height: 1.6;
            color: #f0ffff;
            margin-bottom: 2rem;
        }

        .get-started {
            display: inline-block;
            background: #21899a;
            color: #dffcff;
            padding: 0.9rem 1.6rem;
            border-radius: 8px;
            font-size: 1.2rem;
            box-shadow: 0 0 18px rgba(0,255,255,0.35);
        }

        .auth-card {
            width: 420px;
            padding: 2.2rem;
            border-radius: 22px;
            background: rgba(4, 25, 31, 0.88);
            border: 1px solid rgba(120, 255, 255, 0.22);
            box-shadow: 0 0 35px rgba(0,255,255,0.18);
        }

        .auth-title {
            text-align: center;
            font-size: 3rem;
            color: #8ffcff;
            text-shadow: 0 0 18px rgba(0,255,255,0.7);
            font-family: cursive;
            margin-bottom: 0.3rem;
        }

        .auth-subtitle {
            text-align: center;
            color: #9fbfc5;
            font-size: 1.15rem;
            margin-bottom: 1.8rem;
        }

        .stTextInput label,
        .stFileUploader label,
        .stSelectbox label,
        .stTextArea label {
            color: #bffcff !important;
            font-size: 1.05rem !important;
        }

        .stTextInput input,
        .stTextArea textarea {
            background-color: #061a20 !important;
            color: white !important;
            border: 1px solid rgba(120,255,255,0.35) !important;
            border-radius: 10px !important;
            box-shadow: 0 0 12px rgba(0,255,255,0.12);
        }

        .stSelectbox div[data-baseweb="select"] {
            background-color: #061a20 !important;
            color: white !important;
        }

        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, #1c8fa1, #27a9ba);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.75rem;
            font-size: 1.1rem;
            font-weight: 700;
            box-shadow: 0 0 18px rgba(0,255,255,0.35);
        }

        .switch-text {
            text-align: center;
            margin-top: 1.2rem;
            color: #b7cbd0;
        }

        .dashboard-card {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(120,255,255,0.18);
            padding: 1.5rem;
            border-radius: 18px;
            box-shadow: 0 0 24px rgba(0,255,255,0.10);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    languages = {
    "English": "en",
    "Hindi":"hi",
    "Telugu":"te"
   
   
}

    def login_page():
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown(
                """
                <div class="left-section">
                    <h1>Welcome to<br>VoiceSphere AI</h1>
                    <p>Transform text into your cloned voice. Upload a voice sample, provide your script, and let our model create life-like audio speech.</p>
                    <div class="features">
                        - Few-Shot Voice Cloning<br>
                        - Natural Intonation<br>
                        - Multilingual Speech Generation
                    </div>
                    <div class="get-started">Getting Started</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown('<div class="auth-card">', unsafe_allow_html=True)
            st.markdown('<div class="auth-title">Login Here</div>', unsafe_allow_html=True)

            full_name = st.text_input("Full Name", placeholder="Enter your full name")
            password = st.text_input("Password", type="password", placeholder="Enter password")

            if st.button("Login", type="primary"):
                ok, profile = authenticate_user(full_name, password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.current_user = sanitize_name(full_name)
                    st.session_state.profile = profile
                    st.session_state.page = "dashboard"
                    st.rerun()
                else:
                    st.error("Invalid full name or password.")

            st.markdown('<div class="switch-text">Don’t have an account?</div>', unsafe_allow_html=True)
            if st.button("Sign up here"):
                st.session_state.page = "signup"
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    def signup_page():
        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown(
                """
                <div class="left-section">
                    <h1>Get Started with<br>VoiceSphere AI</h1>
                    <p>Create your account to unlock the power of voice cloning. Define your voice profile and start transforming text into life-like speech.</p>
                    <div class="features">
                        - Secure Voice Data<br>
                        - Personalized Voice Profiles<br>
                        - Full Synthesis Control
                    </div>
                    <div class="get-started">Getting Started</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown('<div class="auth-card">', unsafe_allow_html=True)
            st.markdown('<div class="auth-title">Create Account</div>', unsafe_allow_html=True)
            st.markdown('<div class="auth-subtitle">Create your profile</div>', unsafe_allow_html=True)

            full_name = st.text_input("Full Name", placeholder="Enter your full name", key="signup_name")
            password = st.text_input("Create Password", type="password", placeholder="Choose password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm password", key="signup_confirm")

            selected_language = st.selectbox(
                "Preferred Output Language",
                list(languages.keys()),
                index=0,
                key="signup_language",
            )

            voice_input_method = st.radio("Choose Voice Sample Method",
            ["Upload Voice Sample", "Record Voice Directly"],
            horizontal=True,
            key="voice_input_method",)


            voice_sample = None
            if voice_input_method == "Upload Voice Sample":
                voice_sample = st.file_uploader(
                    "Upload Voice Sample",
                    type=["wav", "mp3", "m4a", "ogg"],
                    help="Upload a clear 5–15 second sample with only one speaker.",
                    key="signup_voice_upload",
                )

            else:
                voice_sample = st.audio_input(
                    "Record Your Voice Sample",
                    key="signup_voice_record",
                )

            if st.button("Create Account & Start Cloning", type="primary"):
                if not full_name.strip():
                    st.error("Enter your full name.")
                elif not password:
                    st.error("Enter password.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                elif voice_sample is None:
                    st.error("Upload your voice sample.")
                else:
                    ok, message = create_account(
                        username=full_name,
                        password=password,
                        language_code=languages[selected_language],
                        uploaded_voice=voice_sample,
                    )

                    if ok:
                        st.success("Account created successfully. Please login.")
                        st.session_state.page = "login"
                        st.rerun()
                    else:
                        st.error(message)

            st.markdown('<div class="switch-text">Already have an account?</div>', unsafe_allow_html=True)
            if st.button("Login here"):
                st.session_state.page = "login"
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

    def dashboard_page():
        profile = st.session_state.profile

        st.sidebar.success(f"Logged in as {profile['username']}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.current_user = None
            st.session_state.profile = None
            st.session_state.page = "login"
            st.rerun()

        st.markdown(
            f"""
            <div class="dashboard-card">
                <h1>🎙️ VoiceSphere AI Dashboard</h1>
                <p>Welcome, <b>{profile['username']}</b>. Enter text, select language, and generate speech in your saved voice.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")

        col1, col2 = st.columns([1.2, 1])

        with col1:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Generate Cloned Speech")

            text_input = st.text_area(
                "Enter Text",
                height=180,
                placeholder="Type the text you want to convert into cloned speech...",
            )

            default_lang_name = next(
                (name for name, code in languages.items() if code == profile["language_code"]),
                "Hindi",
            )

            selected_language = st.selectbox(
                "Output Language",
                list(languages.keys()),
                index=list(languages.keys()).index(default_lang_name),
            )

            if st.button("Generate Speech", type="primary"):
                if not text_input.strip():
                    st.error("Enter text first.")
                else:
                    try:
                        with st.spinner("Generating speech in your saved voice..."):
                            output_path = generate_speech(
                                text=text_input,
                                speaker_wav=profile["voice_sample"],
                                language_code=languages[selected_language],
                                output_prefix=profile["username"],
                            )

                        st.success("Speech generated successfully.")
                        st.audio(str(output_path), format="audio/wav")

                        with open(output_path, "rb") as audio_file:
                            st.download_button(
                                label="Download Generated Audio",
                                data=audio_file,
                                file_name=Path(output_path).name,
                                mime="audio/wav",
                            )

                    except Exception as e:
                        st.error(f"Generation failed: {e}")

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Saved Voice Profile")
            st.write(f"**Full Name:** {profile['username']}")
            st.write(f"**Preferred Language:** `{profile['language_code']}`")
            st.write(f"**Voice Sample:** `{profile['voice_sample']}`")
            st.info("Your uploaded voice sample is used to extract speaker identity for voice cloning.")
            st.markdown("</div>", unsafe_allow_html=True)

    if st.session_state.logged_in:
        dashboard_page()
    else:
        if st.session_state.page == "signup":
            signup_page()
        else:
            login_page()

def run_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Voice Cloning Portal CLI fallback. Use this when Streamlit is unavailable."
    )
    subparsers = parser.add_subparsers(dest="command")

    create_cmd = subparsers.add_parser("create-account", help="Create a user account with a saved voice sample.")
    create_cmd.add_argument("--username", required=True)
    create_cmd.add_argument("--password", required=True)
    create_cmd.add_argument("--language", default=DEFAULT_LANGUAGE_CODE)
    create_cmd.add_argument("--voice", required=True, help="Path to the user voice sample.")

    gen_cmd = subparsers.add_parser("generate", help="Generate speech using a saved user profile.")
    gen_cmd.add_argument("--username", required=True)
    gen_cmd.add_argument("--password", required=True)
    gen_cmd.add_argument("--text", required=True)
    gen_cmd.add_argument("--language", default="")

    subparsers.add_parser("self-test", help="Run built-in tests.")

    args = parser.parse_args(argv)

    if args.command == "create-account":
        voice_path = Path(args.voice)
        if not voice_path.exists():
            print(f"Voice sample not found: {voice_path}")
            return 1
        ok, message = create_account(
            username=args.username,
            password=args.password,
            language_code=args.language,
            uploaded_voice=UploadedFileAdapter(voice_path),
        )
        print(message)
        return 0 if ok else 1

    if args.command == "generate":
        ok, profile = authenticate_user(args.username, args.password)
        if not ok or profile is None:
            print("Invalid username or password.")
            return 1
        language_code = args.language or profile["language_code"]
        try:
            output_path = generate_speech(
                text=args.text,
                speaker_wav=profile["voice_sample"],
                language_code=language_code,
                output_prefix=profile["username"],
            )
        except RuntimeError as exc:
            print(exc)
            return 1
        print(f"Generated file: {output_path}")
        return 0

    if args.command == "self-test":
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(AppTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1

    parser.print_help()
    return 0


class AppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path(tempfile.mkdtemp(prefix="voice_portal_test_"))
        self.paths = AppPaths(
            base_dir=self.temp_root,
            output_dir=self.temp_root / "out",
            profile_dir=self.temp_root / "profiles",
            temp_dir=self.temp_root / "temp",
        )
        self.paths.output_dir.mkdir(exist_ok=True)
        self.paths.profile_dir.mkdir(exist_ok=True)
        self.paths.temp_dir.mkdir(exist_ok=True)
        self.voice_file = self.temp_root / "sample.wav"
        self.voice_file.write_bytes(b"RIFFdemo")

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_sanitize_name(self) -> None:
        self.assertEqual(sanitize_name("A User!"), "A_User")
        self.assertEqual(sanitize_name(""), "user")

    def test_hash_password_is_stable(self) -> None:
        self.assertEqual(hash_password("abc"), hash_password("abc"))
        self.assertNotEqual(hash_password("abc"), hash_password("abcd"))

    def test_create_and_authenticate_account(self) -> None:
        ok, message = create_account(
            username="tester",
            password="secret",
            language_code="hi",
            uploaded_voice=UploadedFileAdapter(self.voice_file),
            paths=self.paths,
        )
        self.assertTrue(ok, message)
        auth_ok, profile = authenticate_user("tester", "secret", paths=self.paths)
        self.assertTrue(auth_ok)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertEqual(profile["language_code"], "hi")
        self.assertTrue(Path(profile["voice_sample"]).exists())

    def test_duplicate_username_fails(self) -> None:
        create_account(
            username="tester",
            password="secret",
            language_code="hi",
            uploaded_voice=UploadedFileAdapter(self.voice_file),
            paths=self.paths,
        )
        ok, message = create_account(
            username="tester",
            password="secret",
            language_code="hi",
            uploaded_voice=UploadedFileAdapter(self.voice_file),
            paths=self.paths,
        )
        self.assertFalse(ok)
        self.assertIn("exists", message)


if __name__ == "__main__":
    if STREAMLIT_AVAILABLE and any("streamlit" in arg.lower() for arg in sys.argv):
        run_streamlit_app()
    elif STREAMLIT_AVAILABLE and Path(sys.argv[0]).name.startswith("streamlit"):
        run_streamlit_app()
    elif STREAMLIT_AVAILABLE and "streamlit" in sys.modules:
        run_streamlit_app()
    else:
        raise SystemExit(run_cli(sys.argv[1:]))
