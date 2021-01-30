import fs from "fs";

export const readJSON = (filename: string) => {
  return new Promise<any>((resolve, reject) => {
    fs.readFile(filename, "utf8", (err, data) => {
      if (err) return reject(err);
      resolve(JSON.parse(data));
    });
  });
};

export const writeJSON = (filename: string, data: any) => {
  return new Promise<void>((resolve, reject) => {
    fs.writeFile(filename, JSON.stringify(data, null, 2), (err) => {
      if (err) return reject(err);
      resolve();
    });
  });
};