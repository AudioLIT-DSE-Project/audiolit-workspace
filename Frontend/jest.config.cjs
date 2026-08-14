/** @type {import('jest').Config} */
module.exports = {
  preset: "ts-jest",
  testEnvironment: "jest-environment-jsdom",
  roots: ["<rootDir>/src"],
  testMatch: ["**/*.test.tsx", "**/*.test.ts"],
  setupFilesAfterEnv: ["<rootDir>/src/tests/setupTests.ts"],
  moduleNameMapper: {
    // Mirror vite.config.ts's "@" -> "./src" alias so tests can import
    // components the same way the app does.
    "^@/(.*)$": "<rootDir>/src/$1",
    // Vite/Webpack-style static asset imports have no meaning to Jest.
    "\\.(css|less|scss)$": "identity-obj-proxy",
  },
  transform: {
    "^.+\\.(t|j)sx?$": [
      "ts-jest",
      {
        tsconfig: {
          jsx: "react-jsx",
          esModuleInterop: true,
        },
      },
    ],
  },
};
