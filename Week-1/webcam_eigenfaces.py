#!/usr/bin/env python3

"""Standalone educational Eigenfaces webcam enrollment and recognition app.



English: Train PCA from Yale_Faces_Images/person_XX/*.png, enroll consented

participants, and run a webcam nearest-neighbor demonstration.

日本語: Yale顔画像からPCAを学習し、Webカメラで人物を登録・照合します。

"""

from __future__ import annotations



import argparse

import csv

from dataclasses import dataclass

from pathlib import Path

import re

import sys

import time
from datetime import datetime
import shutil



import cv2

import joblib

import numpy as np

from sklearn.decomposition import PCA

from sklearn.neighbors import NearestNeighbors



ROOT = Path(__file__).resolve().parent

DEFAULT_DATA = ROOT / 'Yale_Faces_Images'

MODEL_DIR = ROOT / 'saved_model'

MODEL_FILE = MODEL_DIR / 'eigenfaces.joblib'

GALLERY_FILE = MODEL_DIR / 'gallery.npz'

ENROLL_DIR = ROOT / 'enrolled_faces'
RECORD_DIR = ROOT / 'recordings'

FACE_W, FACE_H = 42, 48

POSES = [

    ('Look straight at the camera', 10, 'front'),

    ('Turn slightly to your left', 5, 'left'),

    ('Turn slightly to your right', 5, 'right'),

    ('Smile naturally', 5, 'smile'),

    ('Look slightly upward', 5, 'up'),

]

TARGET_COUNT = sum(count for _, count, _ in POSES)





def grayscale_image(image_bgr_or_gray: np.ndarray) -> np.ndarray:

    """Return the exact 8-bit grayscale image presented to the PCA model.



    Yale PNGs are loaded as grayscale; webcam crops are BGR and converted.

    Both are resized to (width=42, height=48). No color channels enter PCA.

    """

    if image_bgr_or_gray is None or image_bgr_or_gray.size == 0:

        raise ValueError('Empty input face image.')

    if image_bgr_or_gray.ndim == 3:

        gray = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2GRAY)

    elif image_bgr_or_gray.ndim == 2:

        gray = image_bgr_or_gray

    else:

        raise ValueError('Expected a grayscale or BGR face image.')

    return cv2.resize(gray, (FACE_W, FACE_H), interpolation=cv2.INTER_AREA)





def grayscale_vector(image_bgr_or_gray: np.ndarray) -> np.ndarray:

    """Flatten resized grayscale image to a float32 vector in [0, 1]."""

    return grayscale_image(image_bgr_or_gray).astype(np.float32).reshape(-1) / 255.0





def grayscale_preview(frame: np.ndarray, crop: np.ndarray | None):

    """Show what PCA actually receives, without altering the camera pixels."""

    if crop is None or crop.size == 0:

        return

    h, w = frame.shape[:2]

    preview_h = min(180, max(80, h // 4))

    preview_w = round(preview_h * FACE_W / FACE_H)

    enlarged = cv2.resize(grayscale_image(crop), (preview_w, preview_h),

                          interpolation=cv2.INTER_NEAREST)

    enlarged = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)

    x = max(4, w-preview_w-12)

    y = max(4, h-preview_h-40)

    frame[y:y+preview_h, x:x+preview_w] = enlarged[:min(preview_h,h-y), :min(preview_w,w-x)]

    cv2.rectangle(frame, (x-2,y-2), (x+preview_w+2,y+preview_h+2),

                  (240,240,240), 2)

    cv2.putText(frame, 'PCA input 42x48 GRAY', (max(5,x-16), max(16,y-8)),

                cv2.FONT_HERSHEY_SIMPLEX, .43, (255,255,255), 1, cv2.LINE_AA)





def dataset_paths(dataset_dir: Path):

    if not dataset_dir.is_dir():

        raise FileNotFoundError(

            f'Dataset directory not found: {dataset_dir}\n'

            'Extract Yale_Faces_Images.zip beside this script so that '

            'Yale_Faces_Images/person_01/face_01.png exists.'

        )

    persons = sorted(p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith('person_'))

    if not persons:

        raise FileNotFoundError(f'No person_XX subfolders in {dataset_dir}')

    for person in persons:

        for path in sorted(person.glob('*.png')):

            yield path, person.name





def train(dataset_dir: Path, components: int = 80, force: bool = False) -> None:

    if MODEL_FILE.exists() and not force:

        raise FileExistsError('A PCA model already exists. Add --force to retrain. '

                              'WARNING: this clears existing registrations.')

    paths_and_labels = list(dataset_paths(dataset_dir))

    if not paths_and_labels:

        raise ValueError('No PNG images found. Check the dataset folders.')

    print(f'Loading {len(paths_and_labels)} face images / 顔画像を読み込み中...')

    rows, labels = [], []

    for path, label in paths_and_labels:

        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if img is None:

            print(f'Skipping unreadable file: {path}', file=sys.stderr)

            continue

        rows.append(grayscale_vector(img))

        labels.append(label)

    X = np.stack(rows).astype(np.float32)

    n_components = min(components, X.shape[0]-1, X.shape[1])

    print(f'Training PCA: images={len(X)}, dimensions={X.shape[1]}, components={n_components}')

    pca = PCA(n_components=n_components, svd_solver='randomized', random_state=42)

    Z = pca.fit_transform(X).astype(np.float32)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump({'pca':pca, 'size':(FACE_W,FACE_H), 'dataset_path':str(dataset_dir)}, MODEL_FILE)

    np.savez_compressed(GALLERY_FILE, features=Z, labels=np.asarray(labels, dtype=str))

    print(f'Model saved: {MODEL_FILE}')

    print(f'Gallery saved: {GALLERY_FILE}')

    print(f'Explained variance, first {n_components}: {pca.explained_variance_ratio_.sum():.1%}')

    print('WARNING: Retaining the PCA space does not guarantee reliable webcam recognition.')





def load_system():

    if not MODEL_FILE.exists() or not GALLERY_FILE.exists():

        raise FileNotFoundError('Run the train command first. No saved model/gallery found.')

    model = joblib.load(MODEL_FILE)  # Only load files created by this trusted program.

    with np.load(GALLERY_FILE, allow_pickle=False) as saved:

        features = saved['features'].astype(np.float32)

        labels = saved['labels'].astype(str)

    if features.ndim != 2 or len(features) != len(labels):

        raise ValueError('Invalid gallery file.')

    if features.shape[1] != model['pca'].n_components_:

        raise ValueError('PCA model and gallery disagree. Retrain or restore matching files.')

    return model['pca'], features, labels





def face_detector():

    path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    detector = cv2.CascadeClassifier(path)

    if detector.empty():

        raise RuntimeError(f'Could not load Haar detector: {path}')

    return detector





def find_faces(frame, detector):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5,

                                      minSize=(65, 65))

    return sorted([tuple(map(int,r)) for r in rects], key=lambda r:r[2]*r[3], reverse=True)





def crop_face(frame, rect, extra_x=0.28, extra_y=0.36):

    """Crop around detected face; apply consistent margins, no drawn guide."""

    x,y,w,h=rect

    img_h,img_w=frame.shape[:2]

    cx=x+w/2; cy=y+h/2

    crop_w=w*(1+extra_x); crop_h=h*(1+extra_y)

    x1=max(0,int(cx-crop_w/2)); x2=min(img_w,int(cx+crop_w/2))

    y1=max(0,int(cy-crop_h/2)); y2=min(img_h,int(cy+crop_h/2))

    return frame[y1:y2, x1:x2].copy()





def guide_geometry(frame):

    h,w=frame.shape[:2]

    # Shape roughly follows upright human face, and scales with window size.

    center=(w//2,h//2)

    axes=(max(55,int(w*.15)),max(75,int(h*.34)))

    return center,axes





def within_guide(frame, rect) -> bool:

    center, axes=guide_geometry(frame)

    x,y,w,h=rect

    face_x=x+w/2;face_y=y+h/2

    # Haar box is face region, not whole head. Require center and broadly reasonable size.

    dx=(face_x-center[0])/(axes[0]*0.75)

    dy=(face_y-center[1])/(axes[1]*0.60)

    return dx*dx+dy*dy <= 1 and (axes[0]*0.52 <= w <= axes[0]*2.1)





def overlay(frame, messages, face_rects, status, show_guide=True):

    if show_guide:
        center, axes = guide_geometry(frame)
        cv2.ellipse(
            frame, center, axes,
            0, 0, 360,
            (0, 255, 0), 3
        )

    for rect in face_rects:

        x,y,w,h=rect

        cv2.rectangle(frame,(x,y),(x+w,y+h),(60,255,255),2)

    # Black translucent bar makes text legible; native OpenCV font is ASCII-only.

    h,w=frame.shape[:2]

    top=min(h-1, max(128,30+len(messages)*29))

    shade=frame.copy()

    cv2.rectangle(shade,(0,0),(w,top),(0,0,0),-1)

    cv2.addWeighted(shade,0.64,frame,0.36,0,frame)

    for i,line in enumerate(messages):

        cv2.putText(frame,line,(12,28+27*i),cv2.FONT_HERSHEY_SIMPLEX,

                    min(0.65,max(0.43,w/1100)),(255,255,255),2,cv2.LINE_AA)

    cv2.putText(frame,status,(12,h-18),cv2.FONT_HERSHEY_SIMPLEX,

                0.58,(0,255,255),2,cv2.LINE_AA)





def open_camera(camera_id):

    cap=cv2.VideoCapture(camera_id)

    if not cap.isOpened():

        raise RuntimeError(f'Cannot open webcam {camera_id}. Check permissions and close other apps.')

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)

    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    return cap





def validate_identity(value):

    if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}',value):

        raise ValueError('Identity must contain 1-40 ASCII letters, digits, _ or - (no spaces).')

    if value.startswith('person_'):

        raise ValueError('Choose a different identity, e.g. student_01, to avoid Yale ID collision.')

    return value





def enroll(identity, camera_id=0, min_interval=0.45):

    identity=validate_identity(identity)

    pca,features,labels=load_system()

    if identity in labels:

        raise ValueError(f'Identity {identity} is already registered. Choose a new name.')

    detector=face_detector()

    directory=ENROLL_DIR / identity

    if directory.exists() and any(directory.iterdir()):

        raise FileExistsError(f'{directory} already contains files; choose another identity.')

    directory.mkdir(parents=True,exist_ok=True)

    (directory/'color_crops').mkdir(exist_ok=True)

    (directory/'grayscale_pca').mkdir(exist_ok=True)

    cap=open_camera(camera_id)

    sample_features=[]

    count=0

    last_saved=0.0

    win=f'Enroll / Register: {identity}'

    print('SPACE/S = save    Q/ESC = quit    Pose changes automatically')

    print('Ask for consent before taking and storing face images.')

    try:

        while count<TARGET_COUNT:

            ok,raw=cap.read()

            if not ok:

                print('Webcam read failed.');break

            raw=cv2.flip(raw,1)

            rects=find_faces(raw,detector)

            guide_ok=len(rects)==1 and within_guide(raw,rects[0])

            step_count=0

            for prompt,target,tag in POSES:

                if count<step_count+target:

                    pose_prompt=prompt; this_n=count-step_count;this_target=target;pose_tag=tag

                    break

                step_count+=target

            view=raw.copy()

            msgs=[f'Identity: {identity}  |  Total: {count}/{TARGET_COUNT}',

                  f'Instruction: {pose_prompt}',

                  f'Current pose: {this_n}/{this_target}',

                  'SPACE or S: capture  |  Q or ESC: finish early',

                  'Position face in GREEN oval; one face only.']

            status=('READY: press SPACE' if guide_ok else

                    ('Multiple faces: only one person allowed' if len(rects)>1 else

                     'Move closer / center your face in the oval'))

            overlay(view,msgs,rects,status)

            if len(rects)==1:

                grayscale_preview(view,crop_face(raw,rects[0]))

            cv2.imshow(win,view)

            key=cv2.waitKey(1)&0xff

            if key in (27,ord('q')):

                break

            if key in (32,ord('s')):

                if not guide_ok:

                    print('Not saved: exactly one detected face must be inside the guide.')

                    continue

                if time.monotonic()-last_saved<min_interval:

                    print('Wait briefly before next capture.')

                    continue

                face=crop_face(raw,rects[0])

                if face.size==0:

                    print('Invalid face crop, try again.');continue

                sharp=cv2.Laplacian(cv2.cvtColor(face,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var()

                if sharp<20:

                    print(f'Image looks blurry ({sharp:.1f}); hold still and try again.');continue

                vector=grayscale_vector(face)

                feature=pca.transform(vector[None,:])[0].astype(np.float32)

                base=f'{count+1:02d}_{pose_tag}'

                output=directory/'grayscale_pca'/f'{base}.png'

                color_output=directory/'color_crops'/f'{base}.png'

                # Persist the original color crop and the EXACT normalized PCA input.

                # The green guide and text are never baked into either image.

                gray_exact=grayscale_image(face)

                if not cv2.imwrite(str(output),gray_exact):

                    print(f'Failed saving {output}'); continue

                if not cv2.imwrite(str(color_output),face):

                    output.unlink(missing_ok=True)

                    print(f'Failed saving {color_output}'); continue

                sample_features.append(feature)

                count+=1

                last_saved=time.monotonic()

                print(f'Saved {count}/{TARGET_COUNT}: {output.name}')

    finally:

        cap.release()

        cv2.destroyAllWindows()



    if not sample_features:

        print('No faces saved. Registration was not added to the gallery.')

        return

    # Explicitly permit partial enrollments after quitting, but warn about quality.

    if count<TARGET_COUNT:

        print(f'Incomplete enrollment ({count}/{TARGET_COUNT}) — not adding identity to gallery.')

        print(f'Images remain at {directory}; rerun enrollment with a new ID if needed.')

        return

    new=np.stack(sample_features)

    new_labels=np.repeat(identity,len(new))

    np.savez_compressed(GALLERY_FILE,

                        features=np.concatenate((features,new),axis=0),

                        labels=np.concatenate((labels,new_labels)))

    print(f'Registered {identity} using {len(new)} face images.')

    print(f'Original color crops: {directory / "color_crops"}')

    print(f'PCA grayscale inputs: {directory / "grayscale_pca"}')

    print(f'Gallery: {GALLERY_FILE}')





def recognize(camera_id=0, threshold=None, show_distance=True):
    """Live webcam recognition; R starts and S stops MP4 recording."""
    pca, features, labels = load_system()
    detector = face_detector()
    neighbors = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(features)
    cap = open_camera(camera_id)
    win = 'Eigenfaces Live Recognition'

    # Recording is independent from webcam and recognition.
    video_writer = None
    video_path = None
    recording = False
    fps = 20.0  # Playback FPS. Actual processed frames may arrive more slowly.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    print('R: start recording | S: stop/save recording | Q/ESC: quit')
    print('Predictions are classroom demonstrations, not verified identities.')
    if threshold is None:
        print('No calibrated rejection threshold: an unknown person may be mislabeled.')
    else:
        print(f'Using experimental threshold={threshold:.2f}; calibrate separately.')

    try:
        while True:
            ok, raw = cap.read()
            if not ok:
                print('Could not read camera frame.')
                break
            raw = cv2.flip(raw, 1)
            rects = find_faces(raw, detector)
            results = []
            for x, y, w, h in rects:
                face = crop_face(raw, (x, y, w, h))
                if face.size == 0:
                    continue
                z = pca.transform(grayscale_vector(face)[None, :]).astype(np.float32)
                distances, indices = neighbors.kneighbors(z)
                distance = float(distances[0, 0])
                label = str(labels[indices[0, 0]])
                if threshold is not None and distance > threshold:
                    label = 'UNKNOWN'
                results.append((x, y, w, h, label, distance))

            frame = raw.copy()
            for x, y, w, h, label, distance in results:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = f'{label}  dist={distance:.2f}' if show_distance else label
                cv2.putText(frame, text, (max(0, x), max(22, y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2,
                            cv2.LINE_AA)

            msgs = [
                'PCA + nearest-neighbor: classroom demo',
                'R: start recording | S: stop/save | Q/ESC: quit',
                ('No calibrated unknown rejection (label may be wrong)'
                 if threshold is None else
                 f'Unknown rejection: distance > {threshold:.2f}')
            ]
            overlay(frame, msgs, [], f'Faces detected: {len(rects)}', show_guide=False)
            if rects:
                grayscale_preview(frame, crop_face(raw, rects[0]))

            if recording:
                cv2.putText(frame, 'REC', (max(0, frame.shape[1] - 100), 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 2,
                            cv2.LINE_AA)
                # Always match the writer dimensions to the processed frame.
                video_writer.write(frame)

            cv2.imshow(win, frame)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                break
            elif key == ord('r') and not recording:
                RECORD_DIR.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                video_path = RECORD_DIR / f'recognition_{timestamp}.mp4'
                height, width = frame.shape[:2]
                candidate = cv2.VideoWriter(
                    str(video_path), fourcc, fps, (width, height)
                )
                if not candidate.isOpened():
                    candidate.release()
                    print('ERROR: Unable to start MP4 writer (check codec/path).')
                else:
                    video_writer = candidate
                    recording = True
                    print(f'Recording started: {video_path}')
            elif key == ord('s') and recording:
                video_writer.release()
                video_writer = None
                recording = False
                print(f'Recording saved: {video_path}')
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f'Recording saved on exit: {video_path}')
        cap.release()
        cv2.destroyAllWindows()


def delete_identity(identity: str, delete_photos: bool = False) -> None:
    """Remove an identity's feature vectors without retraining the PCA model.

    By default, enrolled photo files are retained. Explicit --delete-photos
    deletes photos only under enrolled_faces/<identity> after gallery update.
    """
    pca, features, labels = load_system()
    mask = labels != identity
    removed = int(np.count_nonzero(~mask))
    if removed == 0:
        print(f'Identity not found: {identity}')
        return

    kept_features = features[mask]
    kept_labels = labels[mask]

    # Write a temporary archive then replace the gallery to reduce the risk
    # of a partially written gallery if saving fails.
    temp_file = GALLERY_FILE.with_name('gallery_temp.npz')
    try:
        np.savez_compressed(temp_file,
                            features=kept_features,
                            labels=kept_labels)
        temp_file.replace(GALLERY_FILE)
    finally:
        temp_file.unlink(missing_ok=True)

    print(f'Removed identity {identity!r} from gallery ({removed} face features).')
    print('PCA model remains unchanged.')
    directory = ENROLL_DIR / identity
    if delete_photos:
        # Protect against deleting arbitrary paths or Yale dataset folders.
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', identity) or identity.startswith('person_'):
            print('Photo deletion skipped: only webcam-enrolled IDs are eligible.')
        elif directory.is_dir():
            shutil.rmtree(directory)
            print(f'Deleted enrolled photos: {directory}')
        else:
            print('No enrolled photo directory exists for this identity.')
    elif directory.is_dir():
        print(f'Photos retained at: {directory}')
        print('Use --delete-photos to remove them when deleting an enrolled ID.')
    if identity.startswith('person_'):
        print('Note: train --force will restore Yale identities from the dataset.')


def restore_identity(identity: str) -> None:
    """Re-enroll an existing set of grayscale PNG images without retraining PCA.

    Requires files under enrolled_faces/<identity>/grayscale_pca/*.png.
    Each image is preprocessed the same way as in train()/enroll().
    """
    identity = validate_identity(identity)
    pca, features, labels = load_system()
    if identity in labels:
        raise ValueError(f'Identity {identity!r} already exists in gallery. Nothing restored.')

    photo_dir = ENROLL_DIR / identity / 'grayscale_pca'
    if not photo_dir.is_dir():
        raise FileNotFoundError(f'Cannot find stored grayscale images: {photo_dir}')
    paths = sorted(photo_dir.glob('*.png'))
    if not paths:
        raise FileNotFoundError(f'No PNG files found in {photo_dir}')

    vectors = []
    for path in paths:
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f'Cannot read grayscale image: {path}')
        vectors.append(grayscale_vector(gray))

    X = np.stack(vectors).astype(np.float32)
    restored_features = pca.transform(X).astype(np.float32)
    updated_features = np.concatenate((features, restored_features), axis=0)
    updated_labels = np.concatenate((labels, np.repeat(identity, len(paths))))

    # Atomically replace the gallery once all images have been processed.
    temp_file = GALLERY_FILE.with_name('gallery_restore_temp.npz')
    try:
        np.savez_compressed(temp_file, features=updated_features, labels=updated_labels)
        temp_file.replace(GALLERY_FILE)
    finally:
        temp_file.unlink(missing_ok=True)

    print(f'Restored {identity!r}: {len(paths)} face images to gallery.')
    print(f'Source images: {photo_dir}')
    print('PCA model unchanged. Run "status" to verify.')


def status():

    pca,features,labels=load_system()

    uniq,counts=np.unique(labels,return_counts=True)

    print(f'PCA components: {pca.n_components_}; gallery: {len(labels)} images, {len(uniq)} identities')

    for name,n in zip(uniq,counts):

        print(f'  {name:20s} {n:4d} images')





def main():

    p=argparse.ArgumentParser(description='Standalone Yale Eigenfaces + webcam face enrollment')

    sub=p.add_subparsers(dest='command',required=True)

    p_train=sub.add_parser('train',help='Train PCA using Yale_Faces_Images folders')

    p_train.add_argument('--data',type=Path,default=DEFAULT_DATA)

    p_train.add_argument('--components',type=int,default=80)

    p_train.add_argument('--force',action='store_true',help='Overwrite model AND all registrations')

    p_enroll=sub.add_parser('enroll',help='Capture and enroll 30 photos of new face')

    p_enroll.add_argument('--id',required=True,help='e.g., student_01')

    p_enroll.add_argument('--camera',type=int,default=0)

    p_enroll.add_argument('--interval',type=float,default=0.45)

    p_live=sub.add_parser('recognize',help='Show live predicted identity')

    p_live.add_argument('--camera',type=int,default=0)

    p_live.add_argument('--threshold',type=float,default=None,

                        help='EXPERIMENTAL Euclidean rejection threshold; calibrate using separate people')

    sub.add_parser('status',help='List saved PCA gallery identities')
    p_delete = sub.add_parser('delete', help='Remove identity from saved gallery')
    p_delete.add_argument('--id', required=True, help='Identity label to remove')
    p_delete.add_argument('--delete-photos', action='store_true',
                          help='Also delete saved webcam photos for this enrolled ID')
    p_restore = sub.add_parser('restore', help='Re-add an identity from saved grayscale photos')
    p_restore.add_argument('--id', required=True, help='Previously enrolled identity')

    args=p.parse_args()

    try:

        if args.command=='train':

            if args.components<1: raise ValueError('--components must be positive')

            train(args.data.resolve(),args.components,args.force)

        elif args.command=='enroll':enroll(args.id,args.camera,args.interval)

        elif args.command=='recognize':recognize(args.camera,args.threshold)

        elif args.command=='delete':delete_identity(args.id,args.delete_photos)

        elif args.command=='restore':restore_identity(args.id)

        elif args.command=='status':status()

    except (FileNotFoundError,FileExistsError,ValueError,RuntimeError) as e:

        print(f'ERROR: {e}',file=sys.stderr)

        sys.exit(1)





if __name__=='__main__':

    main()
