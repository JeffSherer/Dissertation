import openai
import logging
import os

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Set your API key
openai.api_key = os.getenv('sk-None-EGfQZoLV6YUeDUbi2UT1T3BlbkFJGWklA408YPHEmwGlvCh2')

def get_completion(prompt):
    try:
        # Use the correct method for completions
        response = openai.Completion.create(
            model="text-davinci-003",  # Replace with your model
            prompt=prompt,
            max_tokens=100  # Adjust as needed
        )
        return response.choices[0].text.strip()
    except Exception as e:
        logging.error(f"OpenAI API error occurred: {e}")
    return None

if __name__ == "__main__":
    prompt = "Explain how to use the OpenAI API."
    result = get_completion(prompt)
    if result:
        print(result)
    else:
        logging.info("No result returned.")
