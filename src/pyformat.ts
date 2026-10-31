const printableExceptions = /[\p{C}\p{Z}]/u;

function escapedCodePoint(code: number): string {
  if (code <= 0xff) {
    return `\\x${code.toString(16).padStart(2, "0")}`;
  }
  if (code <= 0xffff) {
    return `\\u${code.toString(16).padStart(4, "0")}`;
  }
  return `\\U${code.toString(16).padStart(8, "0")}`;
}

function pyString(text: string): string {
  const quote = text.includes("'") && !text.includes('"') ? '"' : "'";
  let out = quote;
  for (const char of text) {
    if (char === "\\" || char === quote) {
      out += `\\${char}`;
    } else if (char === "\n") {
      out += "\\n";
    } else if (char === "\r") {
      out += "\\r";
    } else if (char === "\t") {
      out += "\\t";
    } else if (char !== " " && printableExceptions.test(char)) {
      out += escapedCodePoint(char.codePointAt(0) ?? 0);
    } else {
      out += char;
    }
  }
  return out + quote;
}

export function pyRepr(value: unknown): string {
  if (typeof value === "string") {
    return pyString(value);
  }
  if (typeof value === "boolean") {
    return value ? "True" : "False";
  }
  if (typeof value === "number") {
    return String(value);
  }
  if (value === null || value === undefined) {
    return "None";
  }
  if (Array.isArray(value)) {
    return `[${value.map(pyRepr).join(", ")}]`;
  }
  return `{${Object.entries(value)
    .map(([key, item]) => `${pyString(key)}: ${pyRepr(item)}`)
    .join(", ")}}`;
}

export function toPromptJson(value: unknown): string {
  if (Array.isArray(value)) {
    return `[${value.map(toPromptJson).join(", ")}]`;
  }
  if (value !== null && typeof value === "object") {
    return `{${Object.entries(value)
      .map(([key, item]) => `${JSON.stringify(key)}: ${toPromptJson(item)}`)
      .join(", ")}}`;
  }
  return JSON.stringify(value ?? null);
}

function pyTimestamp(iso: string, separator: string): string {
  const date = new Date(iso);
  const text = date.toISOString();
  const milliseconds = date.getUTCMilliseconds();
  const fraction = milliseconds === 0 ? "" : `.${String(milliseconds).padStart(3, "0")}000`;
  return `${text.slice(0, 10)}${separator}${text.slice(11, 19)}${fraction}+00:00`;
}

export function pyIsoformat(iso: string): string {
  return pyTimestamp(iso, "T");
}

export function pyDatetime(iso: string): string {
  return pyTimestamp(iso, " ");
}
