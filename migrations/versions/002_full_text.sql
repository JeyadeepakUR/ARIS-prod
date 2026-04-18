-- Migration 002: Add full_text column to documents table
-- Stores the complete extracted text of each document for use in graph building.
-- Deferred loading keeps it out of normal SELECT * queries.

ALTER TABLE documents ADD COLUMN IF NOT EXISTS full_text TEXT;
