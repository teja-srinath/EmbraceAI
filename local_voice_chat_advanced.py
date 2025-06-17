import sys
import argparse
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from fastrtc import ReplyOnPause, Stream, get_stt_model, get_tts_model
from loguru import logger
from ollama import chat

stt_model = get_stt_model()
tts_model = get_tts_model()

ckpt_path = "counsel-chat-bert-classifier/checkpoint-792"
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(ckpt_path)
    # Initialize pipeline
clf = pipeline(
    "text-classification",
    model = model,
    tokenizer = tokenizer,
    top_k = None
    )

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

SYSTEM_PROMPT = """ You are a compassionate and skilled AI personal psychotherapist called Embrace AI. Your role is to provide supportive, thoughtful, and evidence-based guidance using techniques from Cognitive Behavioral Therapy (CBT), Interpersonal Therapy (IPT), and mindfulness practices. 
Your goal is to deeply understand the user's emotional or psychological concern by gently exploring their thoughts, feelings, behaviors, relationships, and context. You listen and respond non-judgmentally and aim to identify cognitive distortions, interpersonal conflicts, or emotional dysregulation. 
Once the core issue is understood, offer tailored, practical support.
Maintain a warm, respectful, and non-directive tone. 

**IMPORTANT**: Avoid diagnosing unless asked. Use clarifying question ONE per response, affirm the user’s strengths and encourage gradual, manageable progress.

** In extreme cases If the user shows signs of severe distress, self-harm, suicidal ideation, or psychological crisis, gently remind them to seek immediate help from a licensed mental health professional or emergency services.**
**Side Note**: You do not replace a licensed therapist, but serve as a supportive tool to promote self-awareness, resilience, and emotional clarity.
"""

def echo(audio):
    transcript = stt_model.stt(audio)
    logger.debug(f"🎤 Transcript: {transcript}")
    response = chat(
        model="mistral-nemo:12b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": transcript},
        ],
        options={"num_predict": 200},
    )
    response_text = response["message"]["content"]
    logger.debug(f"🤖 Response: {response_text}")
    response_text = classifyprompt(response_text)
    for audio_chunk in tts_model.stream_tts_sync(response_text):
        yield audio_chunk

def generateprompt(top): 
    logger.debug(f" context: {top['label']}")
    prompt = f" **CONTEXT** seems to be experiencing {top['label']} focus your conversation towards it."
    return prompt

def initialiseCClasifier(): 
    prompt = "I can't stop thinking about a painful event from my past."

    clf(prompt[:500])[0] 


def classifyprompt(prompt): 
    
    all_scores = clf(prompt[:500])[0] 

    top = max(all_scores, key=lambda x: x["score"])
    if top['score'] > 0.70:
        prompt = prompt + generateprompt(top)
        
    return prompt

def create_stream():
    return Stream(ReplyOnPause(echo), modality="audio", mode="send-receive")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embrace AI: Your virtual support companion.")
    args = parser.parse_args()

    stream = create_stream()

    initialiseCClasifier()
    logger.info("Launching with Gradio UI...")
    stream.ui.launch()
