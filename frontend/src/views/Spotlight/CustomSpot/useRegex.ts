import { useCallback, useEffect, useState } from "react";

/**
 * @deprecated we are not using regex since the default search is good enough
 * @param query
 * @returns
 */
export const useRegex = (query: string) => {
  const [re, setRe] = useState<RegExp>();
  const [isValid, setIsValid] = useState(false);

  useEffect(() => {
    try {
      const matcher = new RegExp(query, "gi");
      setRe(matcher);
      setIsValid(true);
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
    } catch (_error: unknown) {
      setRe(undefined);
      setIsValid(false);
    }
  }, [query]);

  const match = useCallback(
    (query: string) => {
      if (!re) {
        return false;
      }
      return re.test(query);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [re, query]
  );

  return {
    match,
    isValid,
  };
};
