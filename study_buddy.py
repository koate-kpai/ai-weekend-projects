import json
import time
from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError

# Load secrets and initialize client
load_dotenv()
client = OpenAI()


def api_call_with_retry(
    messages,
    model="gpt-4o-mini",
    temperature=0.7,
    max_retries=3,
    response_format=None,
):
    """Call the Chat Completion API with exponential backoff retries on transient errors."""  # noqa : E501
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except (APIError, RateLimitError) as e:
            print(f"⚠️ API error (attempt {attempt+1}/{max_retries}): {e}")
            if attempt == max_retries - 1:
                raise  # Re‑raise error if final attempt fails
            time.sleep(2**attempt)  # Wait 1s, 2s, 4s


def grade_answer(question_text, user_guess):
    """Evaluate the user's answer using structured JSON mode. Returns (is_correct, explanation)."""  # noqa : E501
    eval_messages = [
        {
            "role": "system",
            "content": (
                "You are a strict grader. Evaluate the user's answer to the multiple-choice question. "  # noqa : E501
                "You MUST respond in valid JSON with exactly two fields:\n"
                '1. "status": exact string value must be either "correct" or "incorrect"\n'  # noqa : E501
                '2. "explanation": a one‑sentence explanation of why.'
            ),
        },
        {
            "role": "user",
            "content": f"Question:\n{question_text}\n\nUser's Answer:\n{user_guess}",  # noqa : E501
        },
    ]

    try:
        feedback_raw = api_call_with_retry(
            eval_messages,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        # Clean JSON parsing without convoluted regex fallbacks
        feedback = json.loads(feedback_raw)
        status = feedback.get("status", "").strip().lower()
        explanation = feedback.get("explanation", "No explanation provided.")

        return (status == "correct"), explanation

    except Exception as e:
        # Simple, elegant fallback if JSON decoding or network completely breaks down  # noqa : E501
        print(f"🔄 Grading system error ({e}). Defaulting to safety check.")
        return False, "Could not verify answer due to a system error."


def get_valid_guess():
    """Validates user input ensuring only acceptable choices are passed to the evaluator."""  # noqa : E501
    while True:
        guess = input("Your answer (A/B/C/D) > ").strip().upper()
        if guess in ("A", "B", "C", "D"):
            return guess
        print("❌ Invalid choice. Please enter A, B, C, or D only.")


def play_study_buddy():
    print("🎓 Welcome to the AI Study Buddy!")
    topic = input(
        "What topic do you want to study today? (e.g., Python, History, Space) > "  # noqa : E501
    ).strip()

    total_questions = 5
    asked_questions = []  # Context memory loop array
    results = []  # Storage for final report file

    print(
        f"\nAwesome! Generating a {total_questions}-question quiz on {topic}...\n"  # noqa : E501
    )

    for i in range(total_questions):
        print(f"--- Question {i+1} of {total_questions} ---")

        # ------- API CALL 1: Generate the question with memory context -------
        question_text = api_call_with_retry(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert tutor. Generate ONE multiple-choice question "  # noqa : E501
                        "with options A, B, C, and D about the given topic. "
                        "Do NOT include the correct answer. Output just the question and options."  # noqa : E501
                    ),
                },
                {
                    "role": "user",
                    "content": f"Topic: {topic}\nPreviously asked questions (DO NOT REPEAT THESE): {asked_questions}",  # noqa : E501
                },
            ],
            temperature=0.7,
        )

        asked_questions.append(question_text)
        print(f"\n{question_text}\n")

        # ---------- User input with localized validation ----------
        user_guess = get_valid_guess()

        # ---------- API CALL 2: Resilient Structured Evaluation ----------
        is_correct, explanation = grade_answer(question_text, user_guess)

        # Print UI feedback
        status_str = "✅ CORRECT" if is_correct else "❌ INCORRECT"
        print(f"\n👨‍🏫 Tutor Feedback: {status_str} — {explanation}\n")
        print("-" * 40)

        # Store session payload data
        results.append(
            {
                "question": question_text,
                "user_answer": user_guess,
                "status": "correct" if is_correct else "incorrect",
                "explanation": explanation,
            }
        )

    # ---------- Final Grade Report Calculation ----------
    correct_count = sum(1 for r in results if r["status"] == "correct")
    print(
        f"\n🎉 Quiz Complete! You scored {correct_count} out of {total_questions}."  # noqa : E501
    )

    if correct_count == total_questions:
        print("Perfect score! You are a genius.")
    elif correct_count >= 3:
        print("Great job! Keep studying.")
    else:
        print("Might need to review that topic a bit more!")

    # ---------- Save persistent history logs ----------
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n💾 Your results have been saved to results.json")


if __name__ == "__main__":
    play_study_buddy()
