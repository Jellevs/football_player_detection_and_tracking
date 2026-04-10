from .unified_splitter import UnifiedSplitter


class JerseySplitter(UnifiedSplitter):
    """
    Jersey-only splitter.

    Reuses all logic from UnifiedSplitter but only activates the jersey signal
    by supplying null teams, so the team signal reference is never established.
    """

    def split_tracklet(self, tracklet, next_available_id):
        jerseys   = tracklet.pred_attributes.get("jerseys", [])
        entropies = tracklet.pred_attributes.get("jersey_entropies", [])

        n = len(tracklet.frames)
        if n == 0:
            return None

        jerseys   = list(jerseys)   + [None] * max(0, n - len(jerseys))
        entropies = list(entropies) + [1.0]  * max(0, n - len(entropies))
        teams      = [None] * n
        team_confs = [0.0]  * n

        switch_indices = self._detect_switches(jerseys, entropies, teams, team_confs)

        if not switch_indices:
            return None

        return self._create_fragments(tracklet, switch_indices, n, next_available_id)
