import sqlite3
import tempfile
from pathlib import Path
from data_labeling.migrate import apply_pending, applied_migrations
with tempfile.TemporaryDirectory() as d:
    db = Path(d) / "t.db"
    md = Path(d) / "mig"
    md.mkdir()
    (md / "001_initial.sql").write_text("CREATE TABLE t1(id INT);")
    (md / "002_add.sql").write_text("CREATE TABLE t2(id INT);")
    r1 = apply_pending(db, md)
    print("1st:", r1)
    print("after 1st _migrations:", applied_migrations(db))
    with sqlite3.connect(str(db)) as c:
        c.execute("DELETE FROM _migrations WHERE name='002_add.sql'")
        c.commit()
    print("after DELETE _migrations:", applied_migrations(db))
    r2 = apply_pending(db, md)
    print("2nd:", r2)
