const filter = (candidate: string, query: string): boolean => {
  return candidate
    .toLocaleLowerCase()
    .includes(query.trim().toLocaleLowerCase());
};

export const filterItems = <T>(
  items: T[],
  serialize: (item: T) => string,
  query: string
): T[] => {
  const out: T[] = [];

  for (const item of items) {
    const serialized = serialize(item);
    if (filter(serialized, query)) {
      out.push(item);
    }
  }

  return out;
};
