"""Guest tools: semantic ranking, exact record lookup, and exhaustive filtering."""

from difflib import SequenceMatcher

from retriever import normalize, normalize_name


def create_guest_tools(catalog, retriever):
    """Bind prepared functions to one catalog without global mutable state."""

    def search_guests(query: str, top_k: int = 5, min_score: float = 0.6) -> dict:
        """Find relevant guest profiles using local semantic embedding similarity.

        Use for interests, background, or topics whose wording may differ from the
        profile. Query in English because the guest profiles and embedding model
        are English. Use find_guests_by_name for incomplete names, not embeddings.
        Return ranked candidates above a configurable similarity cutoff,
        not an exhaustive or verified list. Scores are not probabilities; the default
        cutoff 0.6 is a heuristic for this embedding model, not proof of relevance.
        Read descriptions before claiming a match and give the supporting profile
        fact with its source. Do not use top_k results to count
        all guests meeting a condition; use filter_guests for exact conditions.
        Each result includes its guest ID, original fields, source, and similarity score.

        Example task: Find guests interested in computers and programming.
        <execute>
        result = search_guests(query="computers and programming")
        print(result)
        </execute>
        After observing the result, explain relevant profiles and cite their sources.

        Args:
            query: An English description of the interests or background to find.
            top_k: Maximum number of ranked candidates, between 1 and 20; default 5.
            min_score: Minimum cosine similarity from -1 to 1; default 0.6.
        """
        return retriever.search(query, top_k, min_score)

    def get_guest(name: str) -> dict:
        """Get the complete local record for a named guest without semantic guessing.

        Match the full name ignoring capitalization, whitespace, and a leading
        Dr., Lady, or Sir title. Return a clear not-found result for an unknown name.
        Use this for a person's biography, relationship, or email address.
        Do not infer facts or contacts from general model knowledge.

        Example task: Tell me about Ada Lovelace and give me her contact.
        <execute>
        result = get_guest(name="Ada Lovelace")
        print(result)
        </execute>
        Then answer from the actual record, citing its source.

        Args:
            name: The guest's full name, optionally including an honorific.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Guest name must not be empty.")
        matches = [guest for guest in catalog.guests if normalize_name(guest["name"]) == normalize_name(name)]
        return {"found": bool(matches), "guest": dict(matches[0]) if matches else None}

    def filter_guests(field: str, value: str) -> dict:
        """Return ALL guests matching an exact field value by scanning the whole catalog.

        Comparison ignores case and extra whitespace, but is not substring or
        semantic matching. Supported fields: name, relation, relation_category, email.
        relation contains the original free text. relation_category is a coarse
        category: friend, family, colleague, none, or other. Categories are derived
        by explicit rules or supplied in the snapshot, never inferred by the model.
        For all friends, use relation_category='friend', NOT relation='friend'.
        Return all matches and their exact count; there is no top_k truncation.
        Unknown fields raise an error. Zero matches does not mean the filter failed.

        Example task: List all friends of the host.
        <execute>
        result = filter_guests(field="relation_category", value="friend")
        print(result)
        </execute>
        Then report the actual match_count and cite the records.

        Args:
            field: name, relation, relation_category, or email.
            value: The exact value to match, ignoring case and whitespace.
        """
        if field not in {"name", "relation", "relation_category", "email"}:
            raise ValueError("Filter field must be name, relation, relation_category, or email.")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Filter value must not be empty.")
        if field == "relation_category" and normalize(value) not in {"friend", "family", "colleague", "none", "other"}:
            raise ValueError("Unknown relation category. Use list_guests() to list everyone; 'all' is not a category.")
        matches = [dict(guest) for guest in catalog.guests if normalize(guest[field]) == normalize(value)]
        return {"field": field, "value": value, "exhaustive": True, "match_count": len(matches), "guests": matches}

    def list_guests() -> dict:
        """List ALL guests in the local guestbook with their records and exact count.

        Use for 'Who are my guests?', 'Give me all guests', and any request for the
        whole guest list. Takes NO arguments. Never use a name='name' lookup or a
        relation_category='all' filter to list guests. No embeddings or top_k limit.

        Example task: Who are my guests?
        <execute>
        result = list_guests()
        print(result)
        </execute>
        After the observation, list the returned names and sources.
        """
        return {"exhaustive": True, "match_count": len(catalog.guests),
                "guests": [dict(guest) for guest in catalog.guests]}

    def find_guests_by_name(name: str) -> dict:
        """Find guest names by a fragment or a small spelling mistake, WITHOUT embeddings.

        Use when the user gives an incomplete or approximate name, such as 'Nicol',
        'Nik', or 'Ada'. First scan for case-insensitive name substrings; if none
        match, compare spelling with name tokens and prefixes. Fuzzy results are
        possible people to confirm, not exact identity matches. If several names
        match, list the actual candidates and ask which one the user means.
        An empty guests list means no match; never invent placeholder profiles.

        Example task: Find guest what name is like Nicol.
        <execute>
        result = find_guests_by_name(name="Nicol")
        print(result)
        </execute>
        Then show only actual returned names, explaining any fuzzy spelling match.

        Args:
            name: A name fragment or approximate name with at least two characters.
        """
        if not isinstance(name, str) or len(name.strip()) < 2:
            raise ValueError("Provide at least two characters of a guest's name.")
        query = normalize_name(name)
        matches = [dict(guest) for guest in catalog.guests if query in normalize_name(guest["name"])]
        mode = "substring"
        if not matches:
            mode = "fuzzy_spelling"
            scored = []
            for guest in catalog.guests:
                tokens = normalize_name(guest["name"]).split()
                score = max(SequenceMatcher(None, query, part).ratio()
                            for token in tokens for part in (token, token[:len(query)]))
                if score >= 0.75:
                    scored.append({**guest, "name_similarity": score})
            matches = sorted(scored, key=lambda guest: guest["name_similarity"], reverse=True)
        return {"query": name, "match_type": mode, "match_count": len(matches), "guests": matches,
                "status": "found" if matches else "no_matches"}

    return [list_guests, find_guests_by_name, search_guests, get_guest, filter_guests]

