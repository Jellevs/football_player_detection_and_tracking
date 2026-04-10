from .unified_splitter import UnifiedSplitter


class TeamSplitter(UnifiedSplitter):
    """
    Team-only splitter.

    Reuses all logic from UnifiedSplitter but only activates the team signal
    by supplying null jerseys, so the jersey signal reference is never established.
    """

    def split_tracklet(self, tracklet, next_available_id):
        teams      = tracklet.pred_attributes.get("teams", [])
        team_confs = tracklet.pred_attributes.get("team_confs", [])

        n = len(tracklet.frames)
        if n == 0:
            return None

        jerseys    = [None] * n
        entropies  = [1.0]  * n
        teams      = list(teams)      + [None] * max(0, n - len(teams))
        team_confs = list(team_confs) + [0.0]  * max(0, n - len(team_confs))

        switch_indices = self._detect_switches(jerseys, entropies, teams, team_confs)

        if not switch_indices:
            return None

        return self._create_fragments(tracklet, switch_indices, n, next_available_id)
