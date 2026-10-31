const graphTables = `
CREATE TABLE memories (
  id TEXT PRIMARY KEY NOT NULL,
  text TEXT NOT NULL,
  source TEXT NOT NULL,
  source_id TEXT NOT NULL,
  revision TEXT NOT NULL,
  kind TEXT NOT NULL,
  subject TEXT,
  raw TEXT NOT NULL,
  hash TEXT NOT NULL,
  reference_at TEXT NOT NULL,
  created_at TEXT NOT NULL,
  extracted_at TEXT
);
CREATE UNIQUE INDEX memories_by_hash ON memories (hash);
CREATE INDEX memories_by_created ON memories (created_at);
CREATE INDEX memories_by_reference ON memories (reference_at);
CREATE INDEX memories_by_source ON memories (source, source_id);

CREATE TABLE entities (
  id TEXT PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
  type TEXT NOT NULL,
  summary TEXT NOT NULL DEFAULT '',
  embedding BLOB NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX entities_by_updated ON entities (updated_at);

CREATE TABLE entity_aliases (
  alias TEXT PRIMARY KEY NOT NULL,
  entity_id TEXT NOT NULL REFERENCES entities (id) ON DELETE CASCADE
);
CREATE INDEX entity_aliases_by_entity ON entity_aliases (entity_id);

CREATE TABLE facts (
  id TEXT PRIMARY KEY NOT NULL,
  source_id TEXT NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
  target_id TEXT NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
  relation TEXT NOT NULL,
  fact TEXT NOT NULL,
  embedding BLOB NOT NULL,
  valid_at TEXT,
  invalid_at TEXT,
  expired_at TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX facts_by_source ON facts (source_id);
CREATE INDEX facts_by_target ON facts (target_id);
CREATE INDEX facts_current ON facts (expired_at, created_at);

CREATE TABLE fact_episodes (
  fact_id TEXT NOT NULL REFERENCES facts (id) ON DELETE CASCADE,
  memory_id TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
  PRIMARY KEY (fact_id, memory_id)
);
CREATE INDEX fact_episodes_by_memory ON fact_episodes (memory_id);

CREATE TABLE episode_entities (
  memory_id TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
  entity_id TEXT NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
  PRIMARY KEY (memory_id, entity_id)
);
CREATE INDEX episode_entities_by_entity ON episode_entities (entity_id);

CREATE VIRTUAL TABLE memories_fts USING fts5(text, tokenize = 'porter unicode61');
CREATE TRIGGER memories_fts_insert AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts (rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER memories_fts_delete AFTER DELETE ON memories BEGIN
  DELETE FROM memories_fts WHERE rowid = old.rowid;
END;

CREATE VIRTUAL TABLE entities_fts USING fts5(name, tokenize = 'porter unicode61');
CREATE TRIGGER entities_fts_insert AFTER INSERT ON entities BEGIN
  INSERT INTO entities_fts (rowid, name) VALUES (new.rowid, new.name);
END;
CREATE TRIGGER entities_fts_delete AFTER DELETE ON entities BEGIN
  DELETE FROM entities_fts WHERE rowid = old.rowid;
END;
CREATE TRIGGER entities_fts_update AFTER UPDATE OF name ON entities BEGIN
  DELETE FROM entities_fts WHERE rowid = old.rowid;
  INSERT INTO entities_fts (rowid, name) VALUES (new.rowid, new.name);
END;

CREATE VIRTUAL TABLE facts_fts USING fts5(relation, fact, names, tokenize = 'porter unicode61');
CREATE TRIGGER facts_fts_insert AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts (rowid, relation, fact, names) VALUES (
    new.rowid,
    new.relation,
    new.fact,
    coalesce((SELECT name FROM entities WHERE id = new.source_id), '') || ' ' || coalesce((SELECT name FROM entities WHERE id = new.target_id), '')
  );
END;
CREATE TRIGGER facts_fts_delete AFTER DELETE ON facts BEGIN
  DELETE FROM facts_fts WHERE rowid = old.rowid;
END;

CREATE TABLE IF NOT EXISTS jobs (
  episode_id TEXT PRIMARY KEY,
  attempts INTEGER NOT NULL DEFAULT 0,
  next_at INTEGER NOT NULL,
  error TEXT,
  failed INTEGER NOT NULL DEFAULT 0
);
`;

const graphitiGraph = `
DROP TRIGGER memories_fts_insert;
DROP TRIGGER memories_fts_delete;
DROP TRIGGER entities_fts_insert;
DROP TRIGGER entities_fts_delete;
DROP TRIGGER entities_fts_update;
DROP TRIGGER facts_fts_insert;
DROP TRIGGER facts_fts_delete;
DROP TABLE memories_fts;
DROP TABLE entities_fts;
DROP TABLE facts_fts;
DROP TABLE episode_entities;
DROP TABLE fact_episodes;
DROP TABLE facts;
DROP TABLE entity_aliases;
DROP TABLE entities;

ALTER TABLE memories ADD COLUMN source_type TEXT NOT NULL DEFAULT 'text';
ALTER TABLE memories ADD COLUMN source_description TEXT NOT NULL DEFAULT '';
UPDATE memories SET
  source_type = CASE kind WHEN 'message' THEN 'message' WHEN 'transcript' THEN 'message' WHEN 'record' THEN 'json' ELSE 'text' END,
  source_description = source || ' ' || kind,
  extracted_at = NULL;

CREATE TABLE entities (
  id TEXT PRIMARY KEY NOT NULL,
  name TEXT NOT NULL,
  labels TEXT NOT NULL,
  summary TEXT NOT NULL DEFAULT '',
  name_embedding BLOB,
  created_at TEXT NOT NULL
);
CREATE INDEX entities_by_name ON entities (name);

CREATE TABLE facts (
  id TEXT PRIMARY KEY NOT NULL,
  source_id TEXT NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
  target_id TEXT NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  fact TEXT NOT NULL,
  fact_embedding BLOB,
  valid_at TEXT,
  invalid_at TEXT,
  expired_at TEXT,
  created_at TEXT NOT NULL
);
CREATE INDEX facts_by_source ON facts (source_id);
CREATE INDEX facts_by_target ON facts (target_id);

CREATE TABLE fact_episodes (
  fact_id TEXT NOT NULL REFERENCES facts (id) ON DELETE CASCADE,
  memory_id TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
  PRIMARY KEY (fact_id, memory_id)
);
CREATE INDEX fact_episodes_by_memory ON fact_episodes (memory_id);

CREATE TABLE episode_entities (
  memory_id TEXT NOT NULL REFERENCES memories (id) ON DELETE CASCADE,
  entity_id TEXT NOT NULL REFERENCES entities (id) ON DELETE CASCADE,
  PRIMARY KEY (memory_id, entity_id)
);
CREATE INDEX episode_entities_by_entity ON episode_entities (entity_id);

CREATE VIRTUAL TABLE memories_fts USING fts5(content, source, source_description, tokenize = 'unicode61 remove_diacritics 0');
INSERT INTO memories_fts (rowid, content, source, source_description) SELECT rowid, text, source_type, source_description FROM memories;
CREATE TRIGGER memories_fts_insert AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts (rowid, content, source, source_description) VALUES (new.rowid, new.text, new.source_type, new.source_description);
END;
CREATE TRIGGER memories_fts_delete AFTER DELETE ON memories BEGIN
  DELETE FROM memories_fts WHERE rowid = old.rowid;
END;

CREATE VIRTUAL TABLE entities_fts USING fts5(name, summary, tokenize = 'unicode61 remove_diacritics 0');
CREATE TRIGGER entities_fts_insert AFTER INSERT ON entities BEGIN
  INSERT INTO entities_fts (rowid, name, summary) VALUES (new.rowid, new.name, new.summary);
END;
CREATE TRIGGER entities_fts_delete AFTER DELETE ON entities BEGIN
  DELETE FROM entities_fts WHERE rowid = old.rowid;
END;
CREATE TRIGGER entities_fts_update AFTER UPDATE OF name, summary ON entities BEGIN
  DELETE FROM entities_fts WHERE rowid = old.rowid;
  INSERT INTO entities_fts (rowid, name, summary) VALUES (new.rowid, new.name, new.summary);
END;

CREATE VIRTUAL TABLE facts_fts USING fts5(name, fact, tokenize = 'unicode61 remove_diacritics 0');
CREATE TRIGGER facts_fts_insert AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts (rowid, name, fact) VALUES (new.rowid, new.name, new.fact);
END;
CREATE TRIGGER facts_fts_delete AFTER DELETE ON facts BEGIN
  DELETE FROM facts_fts WHERE rowid = old.rowid;
END;
CREATE TRIGGER facts_fts_update AFTER UPDATE OF name, fact ON facts BEGIN
  DELETE FROM facts_fts WHERE rowid = old.rowid;
  INSERT INTO facts_fts (rowid, name, fact) VALUES (new.rowid, new.name, new.fact);
END;

DELETE FROM jobs;
INSERT INTO jobs (episode_id, next_at) SELECT id, row_number() OVER (ORDER BY reference_at, created_at) FROM memories;
`;

export const migrations: readonly string[] = [graphTables, graphitiGraph];
