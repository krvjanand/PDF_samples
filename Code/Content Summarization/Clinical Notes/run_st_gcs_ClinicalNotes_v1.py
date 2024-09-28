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
import re
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
        system_instructions1 = (
            "As a an expert in document entity extraction, you parse the documents to identify and organize specific entities from diverse sources into structured text format, following detailed guidelines for clarity and completeness.\n"
            "Extract only the following data from the PDF document in a valid JSON format only. Avoid unnecessary quotes, comma, text like json’’’, ‘’’json, etc. in the generated structured output."
        )

        system_instructions2 = (
            "For a Nurse / Health care professional, create a Professional Clinical Summary (2 (TWO) Paragraphs) with max. 1,550 characters limit and bold caption as “Summary from Platform_Source”:  The Platform_source detail is provided in the input PDF document.\n"
            "In the first line of Summary include Age, Gender, Preferred Language & the reason for Hospital visit. Continue the summary paragraph with details in the ORDER of Diagnosis, Presenting Symptoms followed by the important details. Additionally, the summary should include any abnormal test results directly related to member’s condition or situation and require further evaluation / treatment.\n"
            "Add \n text between the TWO paragraphs!!\n"
            "In the second paragraph, summary should include procedures / surgeries or treatments received. At the end of the summary, only include MOST RELEVANT Past Medical History. Don’t list all of them!!\n"
            "Include behavioral health information (Social History) / Smoking / Drinkling / in key bullet points.\n"
            "\n"
            "Example Summary:\n"
            "Summary from UM Portal: 58-year old, Female presented to ED for evaluation of constant right lower quadrant abdominal pain which started 2 weeks prior to admission that worsened over the past 4-5 days. The onset of pain was gradual which was described as “pressure-like”, constant , similar to when she passed kidney stones in the past & rated as 10/10 at its worst. She also complained of intractable nausea & vomiting, diarrhea & blood in the stool which was noticed on a toilet paper after she had a BM, intermittently.\n"
            "\n"
            "Labs showed evidence of metabolic acidosis & leukocytosis, urinalysis was possible for UTI. Urine was sent for culture. IV antiemetics & pain meds were given along with IV Rocephin. CT Abd/pelvis without contrast was obtained due to AKI which was likely related to dehydration. It showed mild diverticulosis of the colon, without change of diverticulitis. Small hiatal hernia, unchanged. Colonoscopy showed presence of Grade 2 hemorrhoids.\n"
            "\n"
            "Social History:  \n"
            "Smoking: Patient was interested in smoking cessation \n"
            "Alcohol: <7 drinks / week"
            "\n"
            "Output formatting: Use double quotes for all keys and string values. Avoid unnecessary comma, single quotes, text like json’’’, ‘’’json, etc in the generated structured output."
        )

        # Load Generative Model
        model1 = GenerativeModel(model_id, system_instruction=system_instructions1)
        model2 = GenerativeModel(model_id, system_instruction=system_instructions2)

        # return model1, model2  # Return the initialized model instance
        return model1, model2  # Return the initialized model instance

    except Exception as e:
        print(f"Error initializing AI Platform and Vertex AI: {e}")
        return None

prompt1 = """
{
    "ANTICIPATED DC PLAN”: ” ”,
    “Patient Home Number #”: “”,
    “Patient Emergency Contact 1 Number #”: “”,
    “Primary Care Physician”: “”,
    “Discharge Plan”: ””,
    “Clinical Comments”: ””
}
"""


prompt2 = """
{
    "Summary from platform_source": " ",
    "Social History": " ",
    "Smoking": " ",
    "Alcohol": " "
}
"""

def generate_content1(pdf_file_uri, prompt, model):
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
        print("Generated text Model_1:")
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
        json_data = json.loads(response.text)

        # Write JSON data to file
        with open(output_json, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"JSON output: {output_json}")
        return json_data

    except Exception as e:
        print(f"Error generating content and saving as JSON: {e}")
        return {"error": True, "message": f"Error generating content 1: {e}"}


def generate_content2(pdf_file_uri, prompt, model):
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
        print("Generated text Model_2:")
        print(response.text)

        generated_text = response.text

        # Determine the input PDF filename from the URI
        input_pdf_filename = os.path.basename(pdf_file_uri)

        # Prepare output JSON filename
        output_json = os.path.splitext(input_pdf_filename)[0] + ".json"

        #Replace curly quotes with standard double quotes
        generated_text = generated_text.replace("“", "\"").replace("”", "\"").replace("‘", "'").replace("’", "'")

        # Remove invalid control characters (e.g., newline and tab characters inside the JSON)
        generated_text = re.sub(r'[\x00-\x1f]+', ' ', generated_text)  # Replace control characters with a space

        # Parse the generated text into a dictionary
        # Assuming special instances when model doesn't likely return a JSON string!
        # json_data = {}
        # for line in generated_text.splitlines():
        #     if ':' in line:
        #         key, value = line.strip().split(':', 1)
        #         json_data[key.strip()] = value.strip()

        # Assume response.text contains the JSON string
        try:
            json_data = json.loads(generated_text)
        except json.JSONDecodeError as decode_error:
            print(f"JSON decode error: {decode_error}")
            return

        # Assume response.text contains the JSON string
        # json_data = json.loads(generated_text)

        # Write JSON data to file
        with open(output_json, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)

        print(f"JSON output: {output_json}")
        return json_data
        # return generated_text

    except Exception as e:
        print(f"Error generating content and saving as JSON: {e}")
        return {"error": True, "message": f"Error generating content 2: {e}"}


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


# def generate_content_and_confidence(gcs_uri1, gcs_uri2, prompt1, prompt2, model1, model2):
def generate_content_and_confidence(gcs_uri2, prompt2, model2):
    # json_response1 = generate_content1(gcs_uri1, prompt1, model1)
    json_response2 = generate_content2(gcs_uri2, prompt2, model2)
    print("json_response2:", json_response2)

    json_output_3 = {}

    # if json_response1:
    #     for key in json_response1:
    #         json_output_3[key] = json_response1[key]
    #
    if json_response2:
        for key in json_response2:
            if key in json_output_3:
                if isinstance(json_output_3[key], list) and isinstance(json_response2[key], list):
                    json_output_3[key].extend(json_response2[key])
                else:
                    json_output_3[key] = json_response2[key]

    # if json_response2:
    #     for key, value in json_response2.items():
    #         if key in json_output_3:
    #             if isinstance(json_output_3[key], list) and isinstance(value, list):
    #                 # If both are lists, extend the existing list
    #                 json_output_3[key].extend(value)
    #             elif isinstance(json_output_3[key], dict) and isinstance(value, dict):
    #                 # If both are dictionaries, merge them recursively
    #                 json_output_3[key] = {**json_output_3[key], **value}
    #             else:
    #                 # Overwrite if they are not compatible (e.g., string vs dict, etc.)
    #                 json_output_3[key] = value
    #         else:
    #             # Add the new key-value pair if not already present
    #             json_output_3[key] = value

    return json_output_3


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

def display_pdf_with_slider(uploaded_file, file_content):
    """Multi-page PDF with a slider for page navigation"""

    pdf_reader = PyPDF2.PdfReader(uploaded_file)

    # get the no. of pages in the PDF
    num_pages = len(pdf_reader.pages)

    # Render all the pages as images
    pdf_images = [render_pdf_as_image(file_content, i) for i in range(1, num_pages + 1)]

    # create a slider for page selection
    page_index = st.slider("Select Page", 1, num_pages, 1)

    if pdf_images[page_index - 1]:
        st.image(pdf_images[page_index - 1], use_column_width=True)
    else:
        st.write("Page does not contain an image")

def main():

    st.set_page_config(
        layout="wide",
        page_title="EazyEnroll",
        page_icon=":gear:",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("Summarize Clinical Records / Notes")

    st.sidebar.success('Get Started!')

    menu = ["About", "Value Proposition", "AI Predict"]
    choice = st.sidebar.selectbox("Menu", menu)

    ind_usa_flag_image = Image.open("C:/Users/Sudhaa/PycharmProjects/pythonProject/.venv/INDUSA.png")
    # st.title('Welcome to EazyEnroll  :flag-in: :flag-us:')

    # st.header("AI Tool: Automate UM Fax Processing & Summarize Clinical Notes")
    st.header("AI Tool: Summarize Clinical Records / Notes")
    st.image(ind_usa_flag_image, width=60, use_column_width=False)

    col1, col2, col3 = st.columns([0.25,0.25, 0.5])

    # Display title in the first column
    with col1:
        # background_color = "#FF0000"
        # background_color = "#7DF9FF"
        # st.markdown(f'<h1 style="background-color: {background_color}; color: black;">Welcome to EazyEnroll</h1>',unsafe_allow_html=True)
        # st.title("Welcome to EazyEnroll")
        col1.image('C:/Users/Sudhaa/PycharmProjects/pythonProject/.venv/patient-medical-report-animation_resized.gif', width=300, use_column_width=True)

    with col2:
        background_color="#7DF9FF"

    with col3:
        col3.image('C:/Users/Sudhaa/PycharmProjects/pythonProject/.venv/Salesforce_Summarization.gif', width=200, use_column_width=True)

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
        # st.subheader("UM Faxes Clinical Records Extraction & clinical Notes Summarization for Case Managers")
        st.subheader("Clinical Notes Summarization for Case Managers")

        # uploaded_file1 = st.file_uploader("Upload UM Fax file", type="pdf")
        uploaded_file2 = st.file_uploader("Upload Clinical Notes PDF file", type="pdf")
        st.divider()
        
        # if uploaded_file1 is not None and uploaded_file2 is not None:
        if uploaded_file2 is not None:

            credentials = service_account.Credentials.from_service_account_file(key_file)

            # Create a unique filename or use the original filename
            # destination_blob_name1 = os.path.join(destination_folder, uploaded_file1.name)

            # Read the file content
            #uploaded_file1 = uploaded_file
            # uploaded_file1.seek(0)
            # file_content1 = uploaded_file1.read()

            # Upload the file to GCP
            # gcs_uri1 = upload_to_gcs(file_content1, bucket_name, destination_blob_name1, credentials)
            # print("gcs_uri_1",gcs_uri1)

            # Upload Clinical Notes File
            # Create a unique filename or use the original filename
            destination_blob_name2 = os.path.join(destination_folder, uploaded_file2.name)

            # Read the file content
            # uploaded_file1 = uploaded_file
            uploaded_file2.seek(0)
            file_content2 = uploaded_file2.read()

            # Upload the file to GCP
            gcs_uri2 = upload_to_gcs(file_content2, bucket_name, destination_blob_name2, credentials)
            print("gcs_uri_2", gcs_uri2)

            # Global variable to store credentials
            global_credentials = None

            project_id = "banded-anvil-426308-i1"
            location = "us-central1"
            model_id = "gemini-1.5-pro-001"
            temp = 0.5

            initialized_model = initialize_vertex_ai(key_file, project_id, location, model_id)
            model1, model2 = initialized_model  # Assume initialized_model is already defined


            #Display PDF as Image and JSON response in two columns
            # st.write("UM Fax_File_Name: ", uploaded_file.name)
            st.write("Clinical Notes_File_Name: ", uploaded_file2.name)

            with st.spinner("Authenticating....."):
                time.sleep(7)
            st.success("Authentication Success!")

            placeholder = st.empty()
            placeholder.text("Reading the Data & Summarizing...")
            time.sleep(26)
            placeholder.success("Summarization Completed!")

            #Clear Placeholder:
            #placeholder.empty()
            # import threading

            #Generate content and confidence score
            # json_output_3 = generate_content_and_confidence(gcs_uri1, gcs_uri2, prompt1, prompt2, model1, model2)
            json_output_3 = generate_content_and_confidence(gcs_uri2, prompt2, model2)
            print("JSON output 3:", json_output_3)
            json_response = generate_content2(gcs_uri2, prompt2, model2)
            # st.empty()  # Clear the placeholder text

            # col1, col2, col3 = st.columns([0.3, 0.3, 0.4])
            col2, col3 = st.columns([0.5, 0.5])

            # with col1:
            #     st.header("UM Fax File")
            #     # uploaded_file.seek(0)  # Reset the file pointer to the start
            #     pdf_image1 = render_pdf_as_image(file_content1)
            #     st.image(pdf_image1, use_column_width=True)

            with col2:
                st.header("Clinical Notes")
                # uploaded_file.seek(0)  # Reset the file pointer to the start
                pdf_image2 = render_pdf_as_image(file_content2)
                st.image(pdf_image2, use_column_width=True)

            with col3:
                st.header("Extracted JSON")
                # st.success(confidence_score)
                # st.write(json_output_3)
                # st.json(json_output_3)
                # st.write(json_response)
                st.json(json_response)
                # st.json(json_response, expanded=True)

                download_name = f"{uploaded_file2.name.split('.')[0]}.json"
                st.download_button \
                        (
                        label="Download JSON",
                        file_name=download_name,
                        # data=json_string,
                        # data=json.dumps(json_output_3, ensure_ascii=False),
                        data=json.dumps(json_response, ensure_ascii=False),
                        mime="application/json",
                    )

        
    elif choice == "About":
        st.title("About EazyEnroll")
        st.subheader("UM Faxes Clinical Records Extraction & Clinical Notes summarization for Case Managers ")
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
