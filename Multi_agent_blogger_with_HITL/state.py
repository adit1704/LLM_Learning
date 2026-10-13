from pydantic import BaseModel, Field

class BlogState(BaseModel):
    topic : str = ""
    audience : str = Field(default="general user")

    research : str = ""
    research_feedback : str = ""

    writer_draft : str = ""
    writer_feedback : str = ""

    final_edit : str = ""

