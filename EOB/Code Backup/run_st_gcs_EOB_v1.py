import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account
from google.cloud import aiplatform

#from google.colab import files
from google.oauth2 import service_account
import google.auth
import google.auth.transport.requests

from google.cloud import aiplatform
from google.cloud import storage
import vertexai
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
    Part,
)

from PIL import Image
from pdf2image import convert_from_path
import fitz
import io
import os
import json
import time
import pandas as pd
# import IPython.display
# from IPython.display import display, Image

# Path to your service account key file
key_file = 'C:/Users/Sudhaa/PycharmProjects/pythonProject/.venv/eazyenroll_vertexai_user.json'

# Google Cloud project ID and bucket name
project_id = 'banded-anvil-426308-i1'
bucket_name = 'eazyenroll_test_gemini'
destination_folder = 'input/'

def upload_to_gcs(file_content, bucket_name, destination_blob_name, credentials):
    """Uploads a file to the bucket."""
    storage_client = storage.Client(credentials=credentials, project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(file_content, content_type='application/pdf')
    return f'gs://{bucket_name}/{destination_blob_name}'

def initialize_vertex_ai(key_file, project_id, location, model_id):
    try:

        # Define the required scope
        SCOPES = ['https://www.googleapis.com/auth/cloud-platform']

        # Authenticate using the service account
        global_credentials = service_account.Credentials.from_service_account_file(
            key_file, scopes=SCOPES
        )

        # Set the environment variable for authentication
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_file

        # Verify the credentials
        auth_req = google.auth.transport.requests.Request()
        global_credentials.refresh(auth_req)

        print("Authentication successful.")

        # Initialize AI Platform SDK
        aiplatform.init(credentials=global_credentials, project=project_id, location="us-central1")

        # Initialize Vertex AI SDK
        vertexai.init(credentials=global_credentials, project=project_id, location=location)

        #System Instructions
        system_instructions = (
            "Provide the output in the prompt format EOB Summary : "
            "For a health insurance plan member, Summarise the Explation of Benefits (EOB) document for the breakdown of charges for services received (Doctor and hospital charges) and the date of service. Elaborate on the healthcare Provider who provided the service & types of service, Amount billed by the Provider for the services."
            "Summarize next on the amount insurance company approved, how much insurance company paid and benefits /discounts of the insurance plan.\n"
            "If the details are available, summarize the insurance coverage for the services, , explain the costs for “in network” versus “out of network” services received from the Providers.\n"
            "In the summary, include the amount of the Copays, deductibles, coinsurance, if any, or non-covered charges that the patient is responsible for.\n"
            "Also, if applicable, mention about the EOB denials, that could be due to services not being covered by the plan, insurance terminated, or ineligibility for insurance coverage.\n"
            "In the next paragraph, explain the amount, that has to be paid (Expected Cost) under Patient / member’s responsibility.\n"
            "\n"
            "Example EOB Summary #1\n"
            "The Explanation of Benefits ( EOB) document details the charges for a Preventive Care service received on 04/06/2022. The total cost of the service was $ 599.00 HealthPartners member savings , which is a discount provided by the the insurance company, amounted to $121.45. The medical plan paid $486.15 towards the service. There were no denials for this claim number (add claim number).\n"
            "The Patient/ member’s responsibility is $0.00, meaning the insurance company covered the entire cost of the service.\n"
            "\n"
            "Example EOB Summary #2\n"
            "The explanation of Benefits (EOB) document details the charges for an established patient office or other outpatient visit, Level I service received on 01/01/2019. The total cost of service was $150.00. Plan savings , which is a discount provided by the insurance company (University of Utah Health Plans) . Amounted to $129.70. The insurance company approved $20.30 towards the service. The medical plan paid $10.30 towards the service under the Reason Code “c” which is Contracted Rate Payment. The Copay is $10.00 which the member is responsible to pay for the service. There were no denials for this claim.\n"
            "The patients responsibility is $10.0 meaning the insurance company covered a portion of the service. This is a contracted rate payment.\n"
            "\n"
            "Output formatting: Avoid unnecessary comma, quotes, text like json’’’, ‘’’json, etc in the generated structured output.\n"
        )

        # Load Generative Model
        model = GenerativeModel(model_id, system_instruction=system_instructions)

        return model  # Return the initialized model instance

    except Exception as e:
        print(f"Error initializing AI Platform and Vertex AI: {e}")
        return None


prompt = """
{
    "EOB Summary": " "
}
"""

def generate_content(pdf_file_uri, prompt, model):
    try:
        print("pdf_file_uri:", pdf_file_uri)
        pdf_file = Part.from_uri(pdf_file_uri, mime_type="application/pdf")
        print("After Part:", pdf_file)

        # Prepare contents for model generation
        contents = [pdf_file, prompt]

        print("Print Contents:", contents)

        # Generate content using the model
        response = model.generate_content(contents = contents)

        # Print the generated text (optional)
        print("Generated text:")
        print(response.text)

        # Determine the input PDF filename from the URI
        input_pdf_filename = os.path.basename(pdf_file_uri)

        # Prepare output JSON filename
        output_json = os.path.splitext(input_pdf_filename)[0] + ".json"

        # Assume response.text contains the JSON string
        try:
            json_data = json.loads(response.text)
        except json.JSONDecodeError as decode_error:
            print(f"JSON decode error: {decode_error}")
            return

        # Assume response.text contains the JSON string
        # json_data = json.loads(response.text)

        # Write JSON data to file
        with open(output_json, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"JSON output: {output_json}")
        return json_data

    except Exception as e:
        print(f"Error generating content and saving as JSON: {e}")
        return {"error": True, "message": f"Error generating content: {e}"}

def generate_confidence(pdf_file_uri, gen_prompt, model):
    try:
        print("pdf_file_uri:", pdf_file_uri)
        #Load PDF file
        pdf_file = Part.from_uri(pdf_file_uri, mime_type="application/pdf")

        #prepare contents for model generation
        contents1 = [pdf_file, gen_prompt]
        print("Print contents:", contents1)

        #Generate content using the model
        confidence = model.generate_content(content=contents1)

        #Print Confidence Score
        print("Confidence Score:")
        print(confidence.text)

        confidence_text = confidence.text

        return confidence_text

    except Exception as e:
        print(f"Error generating confidence score and saving to JSON file: {e}")
        return None


def call_vertex_ai_gemini(gcs_uri, prompt):
    """Call Vertex AI Gemini model to generate JSON from PDF."""
    # model_name = "projects/eazyenroll-poc/locations/us-central1/models/gemini-1.5-flash-001"
    model_name = "gemini-1.5-pro-001"

    model = aiplatform.Model(model_name=model_name)
    response = model.predict(instances=[{"content": gcs_uri, "prompt": prompt}])
    return response.predictions[0]

def display_pdf_as_image(pdf_file):
    """Convert PDF to image for display."""
    image = Image.open(io.BytesIO(pdf_file.read()))
    return image

def render_pdf_as_image(pdf_file):
    pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
    pdf_page = pdf_document[0]  # Assuming you want to display the first page

    pdf_image = pdf_page.get_pixmap()
    pil_image = Image.frombytes("RGB", [pdf_image.width, pdf_image.height], pdf_image.samples)

    return pil_image
    
def render_pdf_as_image1(pdf_file, max_width = 720, max_height = None):

    # Create a temporary file-like object
    temp_file = io.BytesIO(pdf_file)

    # Open the PDF using fitz
    doc = fitz.open(temp_file)
    
    # Get the first page (adjust for multi-page PDFs if needed)
    page = doc[0]
    
    # Determine the scaling factor to fit within max dimensions
    zoom_x = max_width / page.media_box.width
    zoom_y = (max_height or float('inf')) / page.media_box.height  # Auto-calculate height if not provided
    zoom = min(zoom_x, zoom_y)

    # Render the page at the calculated zoom factor
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)  # Set alpha=False for better performance

    # Convert the PIL Image to a format suitable for Streamlit
    return pix.asarray()    

def main():

    st.set_page_config(
        layout="wide",
        page_title="EOB Simplification",
        page_icon=":gear:",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("Understand your EOB Documents")

    st.sidebar.success('Get Started!')

    menu = ["About", "Value Proposition", "AI Predict"]
    choice = st.sidebar.selectbox("Menu", menu)

    ind_usa_flag_image = Image.open("C:/Users/Sudhaa/PycharmProjects/pythonProject/.venv/INDUSA.png")
    # st.title('Welcome to EazyEnroll  :flag-in: :flag-us:')

    st.header("AI Tool: summarize Explanation of Benefits (EOB) Documents")
    st.image(ind_usa_flag_image, width=60, use_column_width=False)

    col1, col2, col3 = st.columns([0.25,0.25, 0.5])

    # Display title in the first column
    with col1:
        # background_color = "#FF0000"
        # background_color = "#7DF9FF"
        # st.markdown(f'<h1 style="background-color: {background_color}; color: black;">Welcome to EazyEnroll</h1>',unsafe_allow_html=True)
        col1.image('C:/Users/Sudhaa/PycharmProjects/pythonProject/.venv/chatbot3_small.gif', width=300, use_column_width=True)


    with col2:
        background_color = "#7DF9FF"

    with col3:
        col3.image('C:/Users/Sudhaa/PycharmProjects/pythonProject/.venv/chatbot3_small.gif', width=200, use_column_width=True)

    # with col2:
    #     col3a, col3b = st.columns(2)
    #     with col3a:
    #         st.image(india_flag_image, width=25, use_column_width=False)
    #         st.image(usa_flag_image, width=25, use_column_width=False)
    #
    #     with col3b:
    #         col3b.image('C:/Users/Sudhaa/PycharmProjects/pythonProject/.venv/chatbot3_small.gif')

    if choice == 'Value Proposition':
        value_image = Image.open('C:/Users/Sudhaa/PycharmProjects/pythonProject/.venv/chatbot3_small.gif')
        st.image(value_image, use_column_width=True)

    elif choice == "AI Predict":
        
        # st.title("EazyEnroll")
        st.subheader("EOB Summarization for Plan Members")

        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        st.divider()
        
        if uploaded_file is not None:

            credentials = service_account.Credentials.from_service_account_file(key_file)

            # Create a unique filename or use the original filename
            destination_blob_name = os.path.join(destination_folder, uploaded_file.name)

            # Read the file content
            #uploaded_file1 = uploaded_file
            uploaded_file.seek(0)
            file_content = uploaded_file.read()

            # Upload the file to GCP
            gcs_uri = upload_to_gcs(file_content, bucket_name, destination_blob_name, credentials)
            print("gcs_uri",gcs_uri)

            # Global variable to store credentials
            global_credentials = None

            project_id = "banded-anvil-426308-i1"
            location = "us-central1"
            model_id = "gemini-1.5-pro-001"
            temp = 0.5

            initialized_model = initialize_vertex_ai(key_file, project_id, location, model_id)
            model = initialized_model  # Assume initialized_model is already defined

            #Display PDF as Image and JSON response in two columns
            st.write("Filename: ", uploaded_file.name)

            with st.spinner("Authenticating....."):
                time.sleep(7)
            st.success("Authentication Success!")

            placeholder = st.empty()
            placeholder.text("Extracting Fomr document data...")
            time.sleep(26)
            placeholder.success("Extraction Completed!")

            #Clear Placeholder:
            #placeholder.empty()
            # import threading
            
            def generate_content_wrapper():
                # Call Vertex AI to get JSON
                json_response = generate_content(gcs_uri, prompt, model)
                
                # if json_response:
                #     json_string = json.dumps(json_response) # convert dict to JSON string
                #     gen_prompt = json_string + "Generate a percentage-based Confidence score by comparing the PDF document with the extracted text in the prompt. Return only the 'Confidence Score: percentage value' as an output. No other unwanted text or explanation in the output. Example: Confidence Score: 93%"
                #     confidence_score = generate_confidence(gcs_uri, gen_prompt, model)

                st.empty()  # Clear the placeholder text

                col1, col2 = st.columns(2)

                with col1:
                    st.header("Explanation of Benefits File")
                    # uploaded_file.seek(0)  # Reset the file pointer to the start
                    pdf_image = render_pdf_as_image(file_content)

                    st.image(pdf_image, use_column_width=True)


                with col2:
                    st.header("Extracted JSON")
                    # st.success(confidence_score)
                    st.json(json_response)
                    # st.json(json_response, expanded=True)
                    
                    download_name = f"{uploaded_file.name.split('.')[0]}.json"
                    st.download_button\
                    (
                        label="Download JSON",
                        file_name=download_name,
                        #data=json_string,
                        data=json.dumps(json_response, ensure_ascii=False),
                        mime="application/json",
                    )

            generate_content_wrapper()
            # thread = threading.Thread(target=generate_content_wrapper)
            # thread.start()
        
    elif choice == "About":
        st.title("About EOB Simplification")
        st.subheader("EOB Summarization for Plan Members")
        st.divider()

        st.markdown(
        """
        With EazyEnroll achieve instant Document Entity Extraction and transform Paper forms into structured data.

        This GenAI solution leverages state-of-the-art LLMs and Multimodal capabilties to automatially convert Handwritten / Printed Paper forms into a structured JSON format.
        
        Here's how it works:
        
        1. Upload: Simply upload the Enrollment form Document / Image
        2. GenAI Processing: EazyEnroll analyzes the form layout and extracts all the required entities.
        3. JSON conversion: The form data is transformed into a clean, structured JSON file.
        4. Download button: Download after converted JSON file renders on the screen
        
        \n\n
        Get started and watch as AI transforms your data!
        \n
        **👈 Select AI Predict from the left side menu**   

        """)
        st.write("Thank you for using our app")

    

if __name__ == '__main__':
    main()
