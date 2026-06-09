# study_
from dotenv import load_dotenv
from openai import OpenAI

# Load secrets and initialize client
load_dotenv()
client = OpenAI()


def play_study_buddy():
    print("🎓 Welcome to the AI Study Buddy!")
    topic = input(
        "What topic do you want to study today? (e.g., Python, History, Space) > "  # noqa : E501
    )

    score = 0
    total_questions = 5
    asked_questions = (
        []
    )  # creating a memory bank of questions. So questions are not repeated.

    print(
        f"\nAwesome! Generating a {total_questions}-question quiz on {topic}...\n"  # noqa : E501
    )

    for i in range(total_questions):
        print(f"--- Question {i+1} of {total_questions} ---")

        # ---------------------------------------------------------
        # API CALL 1: Generate the Question
        # ---------------------------------------------------------
        question_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert tutor. Generate ONE multiple-choice question with options A, B, C, and D about the given topic. DO NOT include the correct answer in your response. Just the question and the options.",  # noqa : E501
                },
                {
                    "role": "user",
                    "content": f"Topic: {topic}\nPreviously asked questions (DO NOT REPEAT THESE): {asked_questions}",  # noqa : E501
                },
            ],
            temperature=0.7,  # A little creativity
            # to make the questions interesting
        )

        question_text = question_response.choices[0].message.content
        asked_questions.append(
            question_text
        )  # save questions to the memory bank # noqa : E501
        print(f"\n{question_text}\n")

        # Get the user's guess
        user_guess = input("Your answer (A/B/C/D) > ").strip().upper()

        # ---------------------------------------------------------
        # API CALL 2: Evaluate the Answer
        # ---------------------------------------------------------
        # We pass the original question AND the user's guess to the AI to grade
        eval_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict grader. Evaluate the user's answer to the multiple-choice question. "  # noqa : E501
                    "If they are right, start your response with EXACTLY the word 'CORRECT'. "  # noqa : E501
                    "If they are wrong, start your response with EXACTLY the word 'INCORRECT'. Then, provide a 1-sentence explanation of why.",  # noqa : E501
                },
                {
                    "role": "user",
                    "content": f"Here is the Question:\n{question_text}\n\nHere is the User's Answer:\n{user_guess}",  # noqa : E501
                },
            ],
            temperature=0.0,  # 0.0 temperature makes the AI behave
            # like a strict, predictable robot
        )

        feedback = eval_response.choices[0].message.content
        print(f"\n👨‍🏫 Tutor Feedback: {feedback}\n")
        print("-" * 40)

        # Local Python logic to track the score
        # based on the AI's strict starting word
        if feedback.upper().startswith("CORRECT"):
            score += 1

    # Final Score Output
    print(f"\n🎉 Quiz Complete! You scored {score} out of {total_questions}.")
    if score == total_questions:
        print("Perfect score! You are a genius.")
    elif score >= 3:
        print("Great job! Keep studying.")
    else:
        print("Might need to review that topic a bit more!")


if __name__ == "__main__":
    play_study_buddy()
