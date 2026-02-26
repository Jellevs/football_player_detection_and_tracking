class Tracklet:
    def __init__(self, track_id=None, frames=None, scores=None, bboxes=None, embeddings=None, parent_id=None):
        """
        Create a Tracklet object with ground truth and predicted attributes.
        gt_attributes only retains track_ids — used for fragment pair labeling.
        Role/team/jersey GT fields are removed since the SNP dataset has no
        referee/staff annotations and GT attributes should not drive algorithm decisions.
        """

        self.track_id = track_id
        self.parent_id = parent_id if parent_id is not None else track_id

        self.scores = self.ensure_list(scores)
        self.frames = self.ensure_list(frames)
        self.bboxes = self.ensure_list(bboxes)
        self.embeddings = embeddings if embeddings is not None else []

        # Only gt_track_id is kept — needed for pair labeling in training data generation
        self.gt_attributes = {
            'track_ids': [],
        }

        self.pred_attributes = {
            'jerseys': [],
            'jersey_confs': [],
            'teams': [],
        }

        self.final_jersey = None
        self.final_team = None


    @staticmethod
    def ensure_list(value):
        """ Convert single values or None to lists """
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]


    def append_from_detection(self, frame_idx, detection_idx, detections):
        """ Append data from a detection object. """

        self.frames.append(frame_idx)
        self.bboxes.append(detections.xyxy[detection_idx])
        self.scores.append(detections.confidence[detection_idx])

        # Add ReID embeddings
        if detections.data and 'reid_embedding' in detections.data:
            self.embeddings.append(detections.data['reid_embedding'][detection_idx])

        # Only store gt_track_id — used for pair labeling, not for algorithm decisions
        if detections.data and 'gt_track_id' in detections.data:
            self.gt_attributes['track_ids'].append(detections.data['gt_track_id'][detection_idx])


    def extract(self, start, end):
        """ Extracts a subtrack from the tracklet between two indices """

        subtrack = Tracklet(
            track_id=self.track_id,
            frames=self.frames[start:end + 1],
            scores=self.scores[start:end + 1],
            bboxes=self.bboxes[start:end + 1],
            embeddings=self.embeddings[start:end + 1] if self.embeddings else None,
            parent_id=self.parent_id
        )

        for key in self.pred_attributes:
            if self.pred_attributes[key] is not None and len(self.pred_attributes[key]) > 0:
                subtrack.pred_attributes[key] = self.pred_attributes[key][start:end + 1]

        for key in self.gt_attributes:
            if self.gt_attributes[key] is not None and len(self.gt_attributes[key]) > 0:
                subtrack.gt_attributes[key] = self.gt_attributes[key][start:end + 1]

        return subtrack


    def to_dict(self):
        """ Convert to dictionary format for saving/analysis """

        return {
            'track_id': self.track_id,
            'parent_id': self.parent_id,
            'pred_attributes': self.pred_attributes,
            'gt_attributes': self.gt_attributes,
        }
    

    @property
    def times(self):
        return self.frames
    
    @times.setter
    def times(self, value):
        self.frames = value

    @property
    def features(self):
        return self.embeddings
    
    @features.setter
    def features(self, value):
        self.embeddings = value