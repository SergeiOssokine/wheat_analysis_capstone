import requests

url = "http://localhost:9696/predict"
files = {
    "input": open(
        "../GrainSetData/wheat/test/7_IM/Grainset_wheat_2021-05-13-10-50-06_22_p600s.png",#4_AP/Grainset_wheat_2021-06-14-10-39-16_20_p600s.png",
        "rb",
    )
}
res = requests.post(url, files=files)
print(res.text)
