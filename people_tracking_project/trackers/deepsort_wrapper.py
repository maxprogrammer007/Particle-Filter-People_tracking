# trackers/deepsort_wrapper.py

from deep_sort_realtime.deepsort_tracker import DeepSort

class TrackByDetection:
    def __init__(self, conf_thresh=0.5, img_size=640, iou_thresh=0.5,
                 skip_interval=1, appearance_weight=0.6):
        self.conf_thresh = conf_thresh
        self.img_size = img_size
        self.iou_thresh = iou_thresh
        self.skip_interval = skip_interval
        self.appearance_weight = appearance_weight
        self.frame_id = 0

        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_cosine_distance=1 - appearance_weight,
            nn_budget=100,
            override_track_class=None,
            embedder="mobilenet",
            half=True
        )

    def update(self, detections, frame):
        self.frame_id += 1
        if self.frame_id % self.skip_interval != 0:
            return []

        if not detections:
            return []

        input_dets = []
        for d in detections:
            if len(d) < 5:
                continue
            bbox = [int(d[0]), int(d[1]), int(d[2]), int(d[3])]
            conf = float(d[4])
            input_dets.append((bbox, conf))

        if not input_dets:
            return []

        tracks = self.tracker.update_tracks(input_dets, frame=frame)

        output = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()
            bbox = list(map(int, ltrb))
            output.append((tid, bbox))

        return output
