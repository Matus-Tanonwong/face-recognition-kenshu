from deepface import DeepFace

DeepFace.stream(db_path = "dataset_kenshu", source = 0, enable_face_analysis = False, anti_spoofing = True)