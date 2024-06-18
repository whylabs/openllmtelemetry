import fs from 'fs';
import ini from 'ini';
import path from 'path';

// Define the structure for the configuration
export interface Config {
  whylabs: {
    endpoint: string;
    api_key: string;
  };
  guardrails: {
    endpoint: string;
    api_key: string;
    log_profile: boolean;
  };
}

// Function to read and parse the configuration file used by the python package, 
// you might want to call openllmtelemetry.instrument() in a python notebook to interactively create
// and save this file.
export function readConfig(): Config {
  // Check if HOME environment variable is set
  const homeDirectory = process.env.HOME;
  if (!homeDirectory) {
    throw new Error("Environment variable HOME is not set.");
  }

  const configPath = path.resolve(homeDirectory, '.whylabs/guardrails-config.ini');
  const fileContent = fs.readFileSync(configPath, 'utf-8');
  return ini.parse(fileContent) as Config;
}

