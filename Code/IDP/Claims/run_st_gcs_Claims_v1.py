import streamlit as st
from google.cloud import storage
from google.oauth2 import service_account
from google.cloud import aiplatform

#pip install google-cloud
#pip install google-cloud-storage
#pip install google-cloud-aiplatform
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
key_file = 'C:/Users/dass__000/PycharmProjects/pythonProject/venv/eazyenroll_vertexai_user.json'

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
            "As a an expert in document entity extraction, you parse the documents to identify and organize specific entities from diverse sources into structured JSON formats, following detailed guidelines for clarity and completeness. Avoid unnecessary quotes, comma, text like json’’’, ‘’’json, etc. in the generated structured output.\n"
            "Output formatting: Use double quotes for all keys and string values. Also ensure the Null / null values are enclosed within double quotes. Avoid unnecessary comma, quotes, text like json’’’, ‘’’json, etc in the generated structured output.\n"
        )

        # Load Generative Model
        model = GenerativeModel(model_id, system_instruction=system_instructions)

        return model  # Return the initialized model instance

    except Exception as e:
        print(f"Error initializing AI Platform and Vertex AI: {e}")
        return None


prompt = """
{
  "formtype": " ",
  "1.INSURED PLAN": " ",
  "1a. INSURED'S I.D. NUMBER": " ",
  "2. PATIENT'S NAME": "",
  "3. PATIENT'S DATE OF BIRTH": {
      "YY": " ",
      "MM": " ",
      "DD": " "
    },
  "3. PATIENT'S SEX": " ",
  "4. INSURED'S NAME": " ",  
  "5. PATIENT'S ZIP CODE": " ",
  "5. PATIENT'S TELEPHONE": " ",
  "5. PATIENT'S STATE": " ",
  "5. PATIENT'S CITY": " ",
  "5. PATIENT'S ADDRESS": " ",
  "6. PATIENT RELATIONSHIP TO INSURED": " ",
  "7. INSURED'S ADDRESS": " ",
  "7. INSURED'S CITY": " ",
  "7. INSURED'S STATE": " ",
  "7. INSURED'S ZIP CODE": " ",
  "7. INSURED'S TELEPHONE": " ",
  "9. OTHER INSURED'S NAME": " ",
  "9a. OTHER INSURED'S POLICY OR GROUP": " ",
  "9d. INSURANCE PLAN NAME": " ",
   "10. PATIENT'S CONDITION": [
      {"10a. EMPLOYMENT": " "},
      {"10b. AUTO ACCIDENT": " "},
      {"10c. OTHER ACCIDENT": " "}
	  ],
  "11. INSURED'S POLICY GROUP OR FECA NUMBER": " ",
  "11a. INSURED'S SEX": " ",
  "11c. INSURANCE PLAN NAME": " ",
  "12. PATIENT'S OR AUTHORIZED PERSON'S SIGNATURE": {
       "SIGNED": " ",
       "DATE": " "
    },
  "13. INSURED OR AUTHORIZED PERSON'S SIGNATURE": " ",
  "20. OUTSIDE LAB": " ",
  "20. LAB $CHARGES": " ",
  "21. DIAGNOSIS OR NATURE OF ILLNESS": [
      {
        "ILLNESS A.": " ",
		"ILLNESS B.": " ",
		"ILLNESS C.": " ",
		"ILLNESS D.": " ",
        "ILLNESS E.": " ",
        "ILLNESS F.": " "
      }
    ],
  "22. RESUBMISSION CODE": " ",
  "23. PRIOR AUTHORIZATION NUMBER": " ",
  "24. SERVICES": [
      {
        "YY 1": " ",
        "UNITS 1": " ",
        "PROVIDER ID 1": " ",
        "PLACE OF SERVICE 1": " ",
        "MODIFIER 1": " ",
        "MM 1": " ",
        "Family Plan 1": " ",
        "EMG 1": " ",
        "DIAGNOSIS POINTER 1": " ",
        "DD 1": " ",
        "CPT/HCPCS 1": " ",
        "$CHARGES 1": " "
      },
      {
        "YY 2": " ",
        "UNITS 2": " ",
        "PROVIDER ID 2": " ",
        "PLACE OF SERVICE 2": " ",
        "MODIFIER 2": " ",
        "MM 2": " ",
        "Family Plan 2": " ",
        "EMG 2": " ",
        "DIAGNOSIS POINTER 2": " ",
        "DD 2": " ",
        "CPT/HCPCS 2": " ",
        "$CHARGES 2": " "
      },
      {
        "YY 3": " ",
        "UNITS 3": " ",
        "PROVIDER ID 3": " ",
        "PLACE OF SERVICE 3": " ",
        "MODIFIER 3": " ",
        "MM 3": " ",
        "Family Plan 3": " ",
        "EMG 3": " ",
        "DIAGNOSIS POINTER 3": " ",
        "DD 3": " ",
        "CPT/HCPCS 3": " ",
        "$CHARGES 3": " "
      }
    ],
  "25. FEDERAL TAX I.D. NUMBER": " ",
  "25. FEDERAL IDENTIFICATION": " ",	
  "26. PATIENT'S ACCOUNT NUMBER": " ",
  "27. ACCEPT ASSIGNMENT": " ",
  "28. TOTAL CHARGE": " ",
  "29. AMOUNT PAID": " ",
  "31. PHYSICIAN OR SUPPLIER SIGNATURE": [
       {"SIGNED": " "},
       {"DATE": " "}
    }, 
	"32. SERVICE INFORMATION": [
      {
        "32. SERVICE FACILITY LOCATION": " "
      },
      {
        "32. a. NPI NUMBER": " "
      }
    ],
    "33. BILLING INFORMATION": [
      {
        "33. BILLING PROVIDER PHONE": " "
      },
      {
        "33. BILLING PROVIDER LOCATION": " "
      },
      {
        "33. a. NPI NUMBER": " "
      }
    ]  
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

        response_text = response.text

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
        # return json_data, response_text
        # return response_text

    except Exception as e:
        print(f"Error generating content and saving as JSON: {e}")
        return None

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
        page_title="Automate Paper Claims",
        page_icon=":gear:",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.title("Paper Claims")

    st.sidebar.success('Get Started!')

    menu = ["About", "Value Proposition", "AI Predict"]
    choice = st.sidebar.selectbox("Menu", menu)

    ind_usa_flag_image = Image.open("C:/Users/dass__000/PycharmProjects/pythonProject/venv/INDUSA.png")
    # st.title('Welcome to EazyEnroll  :flag-in: :flag-us:')

    col1, col2 = st.columns([2,1])

    # Display title in the first column
    with col1:
        # background_color = "#FF0000"
        # background_color = "#7DF9FF"
        # st.markdown(f'<h1 style="background-color: {background_color}; color: black;">Welcome to EazyEnroll</h1>',unsafe_allow_html=True)
        st.title("Automate Paper Claims")
        st.image(ind_usa_flag_image, width=60, use_column_width=False)

    with col2:
        col2.image('C:/Users/dass__000/PycharmProjects/pythonProject/venv/Claim-health-insurance-form.gif')

    # with col2:
    #     col3a, col3b = st.columns(2)
    #     col3a, col3b = st.columns(2)
    #     with col3a:
    #         st.image(india_flag_image, width=25, use_column_width=False)
    #         st.image(usa_flag_image, width=25, use_column_width=False)
    #
    #     with col3b:
    #         col3b.image('C:/Users/Sudhaa/PycharmProjects/pythonProject/.venv/chatbot3_small.gif')

    if choice == 'Value Proposition':
        value_image = Image.open('C:/Users/dass__000/PycharmProjects/pythonProject/venv/chatbot3_small.gif')
        st.image(value_image, use_column_width=True)

    elif choice == "AI Predict":
        
        # st.title("EazyEnroll")
        st.subheader("PDF Upload and AI Text Extraction")

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
            placeholder.text("Extracting Form document data...")
            time.sleep(26)
            placeholder.success("Extraction Completed!")

            #Clear Placeholder:
            #placeholder.empty()
            # import threading
            
            def generate_content_wrapper():
                # Call Vertex AI to get JSON
                json_response = generate_content(gcs_uri, prompt, model)
                # json_response, response_text = generate_content(gcs_uri, prompt, model)
                # response_text = generate_content(gcs_uri, prompt, model)
                
                if json_response:
                # if response_text:
                    json_string = json.dumps(json_response) # convert dict to JSON string
                    gen_prompt = json_string + "Generate a percentage-based Confidence score by comparing the PDF document with the extracted text in the prompt. Return only the 'Confidence Score: percentage value' as an output. No other unwanted text or explanation in the output. Example: Confidence Score: 93%"
                    # gen_prompt = response_text + "Generate a percentage-based Confidence score by comparing the PDF document with the extracted text in the prompt. Return only the 'Confidence Score: percentage value' as an output. No other unwanted text or explanation in the output. Example: Confidence Score: 93%"
                    confidence_score = generate_confidence(gcs_uri, gen_prompt, model)

                st.empty()  # Clear the placeholder text

                col1, col2 = st.columns(2)

                with col1:
                    st.header("Claim Form PDF Preview")
                    # uploaded_file.seek(0)  # Reset the file pointer to the start
                    pdf_image = render_pdf_as_image(file_content)

                    st.image(pdf_image, use_column_width=True)

                with col2:
                    st.header("Extracted JSON")
                    # st.success("Confidence Score:", confidence_score)
                    st.json(json_response)
                    # st.json(response_text)
                    # st.json(json_response, expanded=True)
                    
                    download_name = f"{uploaded_file.name.split('.')[0]}.json"
                    st.download_button\
                    (
                        label="Download JSON",
                        file_name=download_name,
                        #data=json_string,
                        data=json.dumps(json_response, ensure_ascii=False),
                        # data=json.dumps(response_text, ensure_ascii=False),
                        mime="application/json",
                    )

            generate_content_wrapper()
            # thread = threading.Thread(target=generate_content_wrapper)
            # thread.start()
        
    elif choice == "About":
        st.title("Automate Paper Claims")
        st.subheader("Extract data from Healthcare Paper Claim Forms to a structured JSON format")
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
