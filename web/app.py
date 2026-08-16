"""Streamlit frontend for Cinescore — Image to Music."""
from __future__ import annotations

import time

import requests
import streamlit as st

API_URL = "http://localhost:8000/api"

st.set_page_config(page_title="Cinescore", page_icon="🎵", layout="wide")

# Header
st.title("🎬 Cinescore")
st.caption(
    "Turn any image into music. "
    "No vocals · ≤ 90 s · Experimental AI-generated output."
)

# Session state defaults
if "job_id" not in st.session_state:
    st.session_state["job_id"] = None
if "data" not in st.session_state:
    st.session_state["data"] = None
if "audio_bytes" not in st.session_state:
    st.session_state["audio_bytes"] = None
if "current_prompt" not in st.session_state:
    st.session_state["current_prompt"] = ""

# Upload
uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png", "webp"],
)


def _poll_for_audio(job_id: str) -> bytes | None:
    """Poll the API until audio is ready or failed. Returns audio bytes or None."""
    for _ in range(120):  # 4 min timeout
        time.sleep(2)
        try:
            resp = requests.get(f"{API_URL}/jobs/{job_id}", timeout=10)
            status_data = resp.json()
        except Exception:
            continue
        if status_data.get("status") == "complete":
            audio_url = f"http://localhost:8000{status_data['audio_url']}"
            return requests.get(audio_url, timeout=30).content
        if status_data.get("status") == "failed":
            st.error(f"Generation failed: {status_data.get('error', 'Unknown error')}")
            return None
    st.error("Generation timed out.")
    return None


if uploaded_file is not None:
    col_img, col_info = st.columns([1, 1])

    with col_img:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Generate
    if st.button("🎵 Generate Music", type="primary", use_container_width=True):
        with st.spinner("Analysing image …"):
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type),
            }
            try:
                resp = requests.post(f"{API_URL}/generate", files=files, timeout=300)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                st.error(f"API error: {exc}")
                st.stop()

        st.session_state["job_id"] = data["job_id"]
        st.session_state["data"] = data
        st.session_state["current_prompt"] = data.get("prompt", "")
        st.session_state["audio_bytes"] = None

        # Poll for audio
        with st.spinner("Generating audio — this may take a minute …"):
            audio_bytes = _poll_for_audio(data["job_id"])
            if audio_bytes:
                st.session_state["audio_bytes"] = audio_bytes

        st.rerun()

    # Display features / spec / audio (from session state)
    data = st.session_state.get("data")
    if data:
        with col_info:
            if data.get("features"):
                feats = data["features"]
                st.subheader("📊 Extracted Features")
                st.write(f"**Caption:** {feats['caption']}")
                for axis, scores in feats["axes"].items():
                    label = axis.replace("_", " ").title()
                    st.write(f"**{label}:** {scores['top']} (conf {scores['confidence']:.2f})")

                colors = feats["color"]["dominant_colors"]
                swatches = "".join(
                    f'<span style="display:inline-block;width:28px;height:28px;'
                    f"background:rgb({r},{g},{b});border-radius:4px;margin:2px;"
                    f'"></span>'
                    for r, g, b in colors
                )
                st.markdown(f"**Palette:** {swatches}", unsafe_allow_html=True)

            if data.get("spec"):
                spec = data["spec"]
                st.subheader("🎼 Music Spec")
                st.write(
                    f"**Genre:** {spec['genre']} · "
                    f"**Tempo:** {spec['tempo_bpm']} BPM · "
                    f"**Key:** {spec['key']} {spec['mode']}"
                )
                st.write(f"**Mood:** {', '.join(spec['mood'])}")
                st.write(f"**Instruments:** {', '.join(spec['instrumentation'])}")

        # Show current prompt
        if st.session_state["current_prompt"]:
            st.subheader("📝 Generated Prompt")
            st.code(st.session_state["current_prompt"])

        # Audio player
        if st.session_state.get("audio_bytes"):
            st.subheader("🎧 Generated Audio")
            st.audio(st.session_state["audio_bytes"], format="audio/wav")
            st.download_button(
                "⬇️ Download WAV",
                st.session_state["audio_bytes"],
                file_name=f"cinescore_{st.session_state['job_id']}.wav",
            )

        # Edit prompt & Regenerate controls
        st.divider()
        edited_prompt = st.text_area(
            "✏️ Edit prompt",
            value=st.session_state["current_prompt"],
        )

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Regenerate from edited prompt", use_container_width=True):
                st.session_state["current_prompt"] = edited_prompt
                st.session_state["audio_bytes"] = None
                with st.spinner("Regenerating with edited prompt …"):
                    try:
                        resp = requests.post(
                            f"{API_URL}/regenerate",
                            data={
                                "job_id": st.session_state["job_id"],
                                "prompt": edited_prompt,
                            },
                            timeout=300,
                        )
                        regen_data = resp.json()
                        audio_bytes = _poll_for_audio(st.session_state["job_id"])
                        if audio_bytes:
                            st.session_state["audio_bytes"] = audio_bytes
                    except Exception as exc:
                        st.error(f"Regeneration error: {exc}")
                st.rerun()

        with col_b:
            if st.button("🎲 Re-roll (new seed, same spec)", use_container_width=True):
                st.session_state["audio_bytes"] = None
                with st.spinner("Re-rolling with a new seed …"):
                    try:
                        resp = requests.post(
                            f"{API_URL}/regenerate",
                            data={
                                "job_id": st.session_state["job_id"],
                                "new_seed": "true",
                            },
                            timeout=300,
                        )
                        audio_bytes = _poll_for_audio(st.session_state["job_id"])
                        if audio_bytes:
                            st.session_state["audio_bytes"] = audio_bytes
                    except Exception as exc:
                        st.error(f"Re-roll error: {exc}")
                st.rerun()
