import time, cv2, pandas, requests, json, numpy as np
import urllib.request
from PIL import Image


class ApiOperator():
    def __init__(self):
        # Code adapted from K. Brian, 2019.
        self.mainURL = "https://crowdspotwebapi.1919280.win.studentwebserver.co.uk/CrowdSpotWebAPI/API/"
        self.jsonHeaders = {'Content-Type': 'application/json'}
        # End of code adapted.

        self.loginTimer = 0
        self.firstLogin = True

        self.streamHeaders = {'Content-Type': 'application/octet-stream'}
        self.jpegHeaders = {"Content-Type": "image/jpeg"}

        self.cameraCode = ""
        self.camera = pandas.DataFrame(
            columns=["cameraID", "locationID", "cameraCode", "operationStatus", "operatingStatus", "cameraName",
                     "cameraDescription", "streamStatus", "marks"])
        self.cameraCordsMarks = pandas.DataFrame(
            columns=["cordID", "cameraID", "markType", "cordXStart", "cordYStart", "cordXEnd", "cordYEnd"]
        )
        self.cameraEnterMarks = []
        self.cameraExitMarks = []

        # Code adapted from PythonTutorial, n.d.
        with open("cameraFile.txt", "r") as file:
            lines = file.readlines()
            self.cameraCode = lines[1]
            if self.cameraCode == "" or self.cameraCode == "\n":
                print(
                    "- No camera code detected. Please put the camera code in the cameraFile.txt \"second line\" then please restart the IoT -")
            else:
                print("- Camera Code Read -")
                print("Code: " + self.cameraCode)
        # End of code adapted.

    def loginCamera(self):
        response = requests.get(self.mainURL + "LoginCamera?cameraCode=" + self.cameraCode,
                                headers=self.jsonHeaders, verify=False)
        if response.status_code == 200:
            self.camera = json.loads(response.content.decode('utf-8'))
            if self.firstLogin:
                print("\n- Camera Login Successful -")
                print("Camera ID: " + str(self.camera["cameraID"]) + " | Camera Name: " + self.camera["cameraName"])
                self.firstLogin = False
            self.loginTimer = time.perf_counter()
            self.extractMarks()
            return True
        else:
            print("\n" + str(json.loads(response.content.decode('utf-8'))))
            return False

    def updateOperatingStatus(self, input: bool):
        if self.camera["cameraID"] is not None:
            if input:
                inputN = 1
            else:
                inputN = 0
            response = requests.put(
                self.mainURL + "ToggleCameraOperating?cameraID=" + str(self.camera["cameraID"]) + "&status=" + str(
                    inputN),
                headers=self.jsonHeaderasds, verify=False)
            print("\n" + str(json.loads(response.content.decode('utf-8'))))

    def postStreamInput(self, image):
        if self.camera["cameraID"] is not None:
            imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.asarray(imageRGB)
            image = Image.fromarray(image)

            # Code Adapted from mmgp (2013) and K. Abhishek (2020)
            image.save("tempStreamImage.jpeg")
            # End of Code Adapted.

            # Code Adapted from P. Martjin (2020).
            files = {"theImage": open("tempStreamImage.jpeg", "rb")}

            response = requests.post(url=self.mainURL + "PostStreamInput?cameraID=" + str(self.camera["cameraID"]),
                                     files=files, verify=False)
            # End of Code Adapted.

            if response.status_code == 200:
                return True
            else:
                return False

    def extractMarks(self):
        if self.camera["cameraID"] is not None:
            response = requests.get(url=self.mainURL + "RetrieveMarks?cameraID=" + str(self.camera["cameraID"]),
                                    headers=self.jsonHeaders, verify=False)
            if response.status_code == 200:
                self.cameraCordsMarks = response.json()
                for i in range(0, len(self.cameraCordsMarks)):
                    if self.cameraCordsMarks[i]["markType"] == 1:
                        self.cameraEnterMarks.append(
                            [
                                [self.cameraCordsMarks[i]["cordXStart"], self.cameraCordsMarks[i]["cordYStart"]],
                                [self.cameraCordsMarks[i]["cordXEnd"], self.cameraCordsMarks[i]["cordYEnd"]]
                            ]
                        )
                    else:
                        self.cameraExitMarks.append(
                            [
                                [self.cameraCordsMarks[i]["cordXStart"], self.cameraCordsMarks[i]["cordYStart"]],
                                [self.cameraCordsMarks[i]["cordXEnd"], self.cameraCordsMarks[i]["cordYEnd"]]
                            ]
                        )
            else:
                print("No marks Detected")

    def countPerson(self, status):
        statusC = ""
        if status:
            statusC = "true"
        else:
            statusC = "false"
        response = requests.post(url=self.mainURL + "CameraUpdateRecordOfLocation?cameraID=" + str(
            self.camera["cameraID"]) + "&count=" + statusC,
                                 headers=self.jsonHeaders, verify=False)
        if response.status_code == 200:
            if status:
                print("Person Count +1")
            else:
                print("Person Count -1")
        else:
            print("Update Person Count Failed!")
