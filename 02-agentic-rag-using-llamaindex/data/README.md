# Guest snapshot

`guests.json` is the full train split of the public
[agents-course/unit3-invitees dataset](https://huggingface.co/datasets/agents-course/unit3-invitees),
downloaded on October 13, 2026. The course uses these profiles as a fictional gala
guestbook. Names, relations, descriptions, and emails are preserved verbatim.
Treat contacts as course data, not verified contact information for public figures.

The snapshot contains three guests: Ada Lovelace, Dr. Nikola Tesla, and Marie Curie.
`GuestCatalog` adds record IDs, source references, and `relation_category` at load time.
The two relations containing the word `friend` map to `friend`; `no relation` maps
to `none`. These are explicit normalization rules, not additional source facts.
An explicitly supplied valid `relation_category` overrides the derived category.

To refresh the snapshot, run `python scripts/download_guests.py`. The vector cache
is rebuilt automatically when the resulting records or embedding model change.
