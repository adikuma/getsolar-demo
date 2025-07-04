import os
import json
from dotenv import load_dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

class CustomerJourneySteps(BaseModel):
    process_flow: List[str] = Field(
        description="each step name/description from landing to install"
    )
    required_information: List[str] = Field(
        description="data fields customers must provide"
    )
    wait_times: List[str] = Field(
        description="all time promises, e.g. 'within 48 hours'"
    )
    human_touchpoints: List[str] = Field(
        description="where customers speak to a real person"
    )
    documentation_requirements: List[str] = Field(
        description="documents customers need to submit"
    )


class CommunicationChannels(BaseModel):
    contact_methods: List[str] = Field(
        description="e.g. whatsapp, email, phone"
    )
    response_time_promises: List[str] = Field(
        description="e.g. 'we’ll get back to you in X hours'"
    )
    languages_supported: List[str] = Field(
        description="any multilingual support mentioned"
    )
    business_hours: Optional[str] = Field(
        description="hours of operation"
    )
    escalation_paths: List[str] = Field(
        description="special-purpose emails or contacts"
    )


class CompanyInfo(BaseModel):
    name: str = Field(description="company name")
    headquartered_in: Optional[str] = Field(description="headquarters location")
    founding_year: Optional[int] = Field(description="year founded")
    short_description: Optional[str] = Field(
        description="brief company description"
    )


class WebsiteData(BaseModel):
    customer_journey_steps: CustomerJourneySteps
    communication_channels: CommunicationChannels
    company_info: CompanyInfo


def main():
    with open("../../data/scrape_result.json", "r") as f:
        raw = json.load(f) 

    raw_json_str = json.dumps(raw, indent=2)

    load_dotenv()
    llm = init_chat_model(
        "gemini-2.0-flash", model_provider="google_genai", temperature=0.0, api_key=gemini_api_key
    )

    structured = llm.with_structured_output(
        WebsiteData, method="json_mode"
    )

    result: WebsiteData = structured.invoke(
        f"### here is the full website scrape as json:\n{raw_json_str}"
    )

    print(result.json(indent=2))

    with open("../data/extracted_structured.json", "w") as f:
        f.write(result.json(indent=2))

if __name__ == "__main__":
    main()