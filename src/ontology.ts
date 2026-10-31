export const entityTypes: Readonly<Record<string, string>> = {
  Preference: "User preferences, choices, opinions, or selections (PRIORITIZE over most other types except User/Assistant)",
  Requirement: "Specific needs, features, or functionality that must be fulfilled",
  Procedure: "Standard operating procedures and sequential instructions",
  Location: "Physical or virtual places where activities occur",
  Event: "Time-bound activities, occurrences, or experiences",
  Organization: "Companies, institutions, groups, or formal entities",
  Document: "Information content in various forms (books, articles, reports, etc.)",
  Topic: "Subject of conversation, interest, or knowledge domain (use as last resort)",
  Person: "An individual human referenced in the content",
  Object: "Physical items, tools, devices, or possessions (use as last resort)",
};

export const customExtractionInstructions =
  "The episode starts with a header (Source, Kind, Occurred, Participants, Subject), then a line ---, then the body. " +
  "Header field names are labels, never entities or speakers. The first participant wrote a message body, and first person " +
  "(I, my, me, we) in the body refers to that author.";
