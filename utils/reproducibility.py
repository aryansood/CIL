import os
import re
import uuid
import time
import shutil
from datetime import datetime
from pathlib import Path
import constants as C

class Submission:
    def __init__(self, description="No Description"):
        self.cwd = Path.cwd()
        self.uuid = uuid.uuid4()
        self.description = description
        self.base_path = Path(C.SNAPSHOTS_DIR) / f"run-{self.uuid}"
        self.output_id = 0        
        
        self.ensure_dir(self.base_path)
        self.reproduce_code()
        self.reproduce_attributes()

    def ensure_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def reproduce_code(self):
        code_base_path = self.base_path / "code"
        code_list = [code_path.relative_to(self.cwd) for code_path in Path(self.cwd).rglob("*.py") if not re.match(C.SNAPSHOTS_DIR, str(code_path))]
        
        for code_path in code_list:
            self.ensure_dir((code_base_path / code_path).parent)
            shutil.copy(self.cwd / code_path, code_base_path / code_path)
    
    def reproduce_attributes(self):
        attribute_base_path = self.base_path / "attributes"
        self.ensure_dir(attribute_base_path)
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(attribute_base_path / ".time", 'w') as fh: fh.write(time)
        with open(attribute_base_path / ".description", 'w') as fh: fh.write(self.description)

    def save_submission(self):
        submission_base_path = self.base_path / "submission"
        self.ensure_dir(submission_base_path)
        pass



if __name__ == '__main__':
    Submission("Test run").save_submission()