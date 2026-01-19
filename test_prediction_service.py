import typer
import requests
from typing_extensions import Annotated


def main(image_path: Annotated[str, typer.Option(help="Path to payload image")]):
    url = "http://localhost:9696/predict"
    files = {
        "input": open(
            image_path,
            "rb",
        )
    }
    res = requests.post(url, files=files)
    print(f"Service response: {res.text}")


if __name__ == "__main__":
    typer.run(main)
