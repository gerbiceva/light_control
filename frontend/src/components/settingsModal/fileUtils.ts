import { FileWithPath } from "@mantine/dropzone";
import { $projectName } from "../../globalStore/projectStore";

export const downloadSettings = () => {
  const settings = JSON.stringify("getSaveFile()");
  const blob = new Blob([settings], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const projectName = $projectName.get();

  const a = document.createElement("a");
  a.href = url;
  a.download = `${projectName}.json`;
  document.body.appendChild(a);
  a.click();

  // Clean up
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
};

export const readFile = async <Type>(file: FileWithPath): Promise<Type> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = () => {
      try {
        const result = JSON.parse(reader.result as string) as Type;
        resolve(result);
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
      } catch (_error) {
        reject(new Error("Failed to parse file content as JSON."));
      }
    };

    reader.onerror = () => {
      reject(new Error("Failed to read the file."));
    };

    reader.readAsText(file);
  });
};
