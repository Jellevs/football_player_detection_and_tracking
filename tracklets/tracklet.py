class Tracklet:
    def __init__(self, track_id=None, frames=None, scores=None, bboxes=None, embeddings=None, teams=None, jerseys=None, roles=None, gt_track_ids=None, parent_id=None):
        """
        Create a Tracklet object with ground truth and predicted attributes
        """

        self.track_id = track_id
        self.parent_id = parent_id if parent_id is not None else track_id

        self.scores = self.ensure_list(scores)
        self.frames = self.ensure_list(frames)
        self.bboxes = self.ensure_list(bboxes)
        self.embeddings = embeddings if embeddings is not None else []
        
        self.gt_attributes = {
            'track_ids': [],
            'jerseys': [],
            'teams': [],
            'roles': []
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

    @property
    def times(self):
        """Alias for frames - GTA compatibility"""
        return self.frames

    @property
    def features(self):
        """Alias for embeddings - GTA compatibility"""
        return self.embeddings

    def append_from_detection(self, frame_idx, detection_idx, detections):
        """ Append data from a detection object. """

        self.frames.append(frame_idx)
        self.bboxes.append(detections.xyxy[detection_idx])
        self.scores.append(detections.confidence[detection_idx])
        
        # add ReID embeddings
        if detections.data and 'reid_embedding' in detections.data:

            self.embeddings.append(detections.data['reid_embedding'][detection_idx])
    

        # add GT attributes | predicted attributes get added in attributes.py
        if 'gt_track_id' in detections.data:
            self.gt_attributes['track_ids'].append(detections.data['gt_track_id'][detection_idx])
        if 'gt_jersey' in detections.data:
            self.gt_attributes['jerseys'].append(detections.data['gt_jersey'][detection_idx])
        if 'gt_team' in detections.data:
            self.gt_attributes['teams'].append(detections.data['gt_team'][detection_idx])
        if 'gt_role' in detections.data:
            self.gt_attributes['roles'].append(detections.data['gt_role'][detection_idx])


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
            # 'frames': self.frames,
            # 'bboxes': self.bboxes,
            # 'scores': self.scores,
            # 'embeddings': self.embeddings,
            'pred_attributes': self.pred_attributes,
            'gt_attributes': self.gt_attributes,
            # 'length_track': len(self.frames),
            # 'final_jersey': self.final_jersey,
            # 'final_team': self.final_team,
            # 'final_role': self.final_role
        }


    # --- GTA Compatibility Section ---

    @property
    def times(self):
        """Alias for frames - GTA compatibility"""
        return self.frames

    @times.setter
    def times(self, value):
        """Allows GTA to overwrite/extend frames using the 'times' alias"""
        self.frames = value

    @property
    def features(self):
        """Alias for embeddings - GTA compatibility"""
        return self.embeddings

    @features.setter
    def features(self, value):
        """Allows GTA to overwrite/extend embeddings using the 'features' alias"""
        self.embeddings = value

    # ----------------------------------
    def append_emb(self, emb):
        '''
        Appends a feature vector to the tracklet.

        Args:
            feat (np.array): Feature vector of shape (512,).
        '''
        self.embeddings.append(emb)