import os
import logging
import click
import torch
import utils
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager

# 🔹 Force CPU globally
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.device("cpu")

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template
from utils import get_embeddings
from langchain_chroma import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)
from load_models import (
    load_quantized_model_awq,
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)
from const import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS,
)


def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    Runs strictly on CPU.
    """
    # 🔹 Force CPU inside the function as well
    device_type = "cpu"

    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        elif ".awq" in model_basename.lower():
            model, tokenizer = load_quantized_model_awq(model_id, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    generation_config = GenerationConfig.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.2,
        repetition_penalty=1.15,
        generation_config=generation_config,
        device=-1,  # 🔹 Force CPU in the HF pipeline
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded on CPU ✅")

    return local_llm


def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="llama"):
    """Initializes and returns a retrieval-based Question Answering (QA) pipeline."""
    device_type = "cpu"  # 🔹 Always CPU
    embeddings = get_embeddings(device_type)

    logging.info(f"Loaded embeddings from {EMBEDDING_MODEL_NAME} (CPU Mode)")

    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()

    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt},
        )

    return qa


# 🔹 Default device_type changed from CUDA → CPU
@click.command()
@click.option(
    "--device_type",
    default="cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Forced to CPU for AMD-based systems)",
)
@click.option("--show_sources", "-s", is_flag=True, help="Show sources along with answers (Default is False)")
@click.option("--use_history", "-h", is_flag=True, help="Use history (Default is False)")
@click.option(
    "--model_type",
    default="llama3",
    type=click.Choice(["llama3", "llama", "mistral", "non_llama"]),
    help="Model type, e.g., llama3, llama, mistral, or non_llama",
)
@click.option("--save_qa", is_flag=True, help="Save Q&A pairs to a CSV file (Default is False)")
def main(device_type, show_sources, use_history, model_type, save_qa):
    """Main interactive Q&A loop for SmartAI Bot."""
    device_type = "cpu"  # 🔹 Force CPU at runtime too

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)

    while True:
        query = input("\nEnter a query: ")
        if query.strip().lower() == "exit":
            break

        res = qa.invoke(query)
        answer, docs = res["result"], res["source_documents"]

        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        if show_sources:
            print("----------------------------------SOURCE_DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE_DOCUMENTS---------------------------")

        if save_qa:
            utils.log_to_csv(query, answer)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
