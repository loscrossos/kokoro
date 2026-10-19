import re

def patch_pipeline():
    file = "kokoro/pipeline.py"
    with open(file, "r") as f:
        content = f.read()

    start_str = """    def load_single_voice(self, voice: str):
        if voice in self.voices:
            return self.voices[voice]
        if voice.endswith('.pt'):
            f = voice
        else:
            f = hf_hub_download(repo_id=self.repo_id, filename=f'voices/{voice}.pt')"""
            
    replace_str = """    def load_single_voice(self, voice: str):
        if voice in self.voices:
            return self.voices[voice]
        if voice.endswith('.pt'):
            f = voice
            if not os.path.exists(f):
                if self.lang_code == 'd':
                    raise RuntimeError(f"German voice tensor '{voice}' not found locally. Please download a community German voice checkpoint (e.g., from Hugging Face) into your voices directory or specify a custom voice path.")
                raise FileNotFoundError(f"Voice tensor '{voice}' not found locally.")
        else:
            try:
                f = hf_hub_download(repo_id=self.repo_id, filename=f'voices/{voice}.pt')
            except Exception as e:
                if self.lang_code == 'd':
                    logger.warning(f"Voice '{voice}' not found in {self.repo_id}. Attempting community fallback...")
                    try:
                        f = hf_hub_download(repo_id='cryptomilk/kokoro-german-kerstin', filename=f'voices/{voice}.pt')
                    except Exception:
                        raise RuntimeError(f"German voice tensor '{voice}' not found locally. Please download a community German voice checkpoint (e.g., from Hugging Face) into your voices directory or specify a custom voice path.")
                else:
                    raise e"""
                    
    content = content.replace(start_str, replace_str)
    
    with open(file, "w") as f:
        f.write(content)

def rewrite_test():
    pass

if __name__ == "__main__":
    patch_pipeline()
    print("Patched pipeline.py!")
