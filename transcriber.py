#!/usr/bin/env python3
"""
==============================================================================
 Technical Video Transcriber — Google Colab Edition
 Version: 2.0.0
 
 High-accuracy transcription for technical courses using faster-whisper
 with OpenAI large-v3 model. Features:
   - GPU-accelerated transcription
   - Tech speech-to-symbol conversion (slash→/, dot com→.com, etc.)
   - Number verbalization handling (eight zero eight zero → 8080)
   - Proper noun capitalization (Docker, Kubernetes, Linux, etc.)
   - TurboScribe-style paragraph formatting
   - Anti-hallucination measures
   - Multi-file batch processing with auto-download
 
 Usage in Google Colab:
   1. Upload video files when prompted
   2. Script auto-installs dependencies, transcribes, and downloads results
   
 Or pull from GitHub:
   !wget https://raw.githubusercontent.com/YOUR_USER/colab-transcriber/main/transcriber.py
   %run transcriber.py
==============================================================================
"""

import os
import re
import gc
import time
import shutil
import glob
from pathlib import Path

# ============================================================
# STEP 1: INSTALL DEPENDENCIES (Colab only)
# ============================================================
def install_dependencies():
    """Install required packages in Google Colab."""
    try:
        import faster_whisper
        print("✅ faster-whisper already installed")
    except ImportError:
        print("📦 Installing faster-whisper...")
        os.system("pip install -q faster-whisper")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        else:
            print("⚠️  No GPU detected — transcription will be slow")
    except ImportError:
        print("⚠️  PyTorch not available")

install_dependencies()

import torch
from faster_whisper import WhisperModel


# ============================================================
# STEP 2: PROPER NOUNS DICTIONARY
# ============================================================
PROPER_NOUNS = {
    # === Programming Languages & Runtimes ===
    'javascript': 'JavaScript', 'typescript': 'TypeScript', 'python': 'Python',
    'kotlin': 'Kotlin', 'java': 'Java', 'golang': 'Golang', 'swift': 'Swift',
    'rust': 'Rust', 'ruby': 'Ruby', 'php': 'PHP', 'c sharp': 'C#',
    'c plus plus': 'C++', 'dart': 'Dart', 'scala': 'Scala', 'perl': 'Perl',
    'lua': 'Lua', 'groovy': 'Groovy', 'elixir': 'Elixir', 'haskell': 'Haskell',
    'node.js': 'Node.js', 'nodejs': 'Node.js', 'node js': 'Node.js',
    'deno': 'Deno', 'bun': 'Bun',
    
    # === Frameworks & Libraries ===
    'react': 'React', 'react native': 'React Native', 'angular': 'Angular',
    'vue': 'Vue', 'vue.js': 'Vue.js', 'next.js': 'Next.js', 'nextjs': 'Next.js',
    'nuxt': 'Nuxt', 'svelte': 'Svelte', 'django': 'Django', 'flask': 'Flask',
    'fastapi': 'FastAPI', 'express': 'Express', 'express.js': 'Express.js',
    'spring': 'Spring', 'spring boot': 'Spring Boot', 'hibernate': 'Hibernate',
    'jetpack compose': 'Jetpack Compose', 'material design': 'Material Design',
    'tailwind': 'Tailwind', 'tailwind css': 'Tailwind CSS',
    'bootstrap': 'Bootstrap', 'jquery': 'jQuery', 'redux': 'Redux',
    'flutter': 'Flutter', 'swiftui': 'SwiftUI',
    'retrofit': 'Retrofit', 'okhttp': 'OkHttp', 'dagger': 'Dagger',
    'hilt': 'Hilt', 'koin': 'Koin', 'ktor': 'Ktor',
    'room': 'Room', 'room database': 'Room Database',
    'camerax': 'CameraX', 'workmanager': 'WorkManager',
    'coroutines': 'Coroutines', 'livedata': 'LiveData',
    'viewmodel': 'ViewModel', 'datastore': 'DataStore',
    
    # === Cloud & Infrastructure ===
    'aws': 'AWS', 'amazon web services': 'Amazon Web Services',
    'ec2': 'EC2', 'ecs': 'ECS', 's3': 'S3', 'rds': 'RDS',
    'lambda': 'Lambda', 'cloudfront': 'CloudFront', 'cloudwatch': 'CloudWatch',
    'dynamodb': 'DynamoDB', 'sqs': 'SQS', 'sns': 'SNS', 'iam': 'IAM',
    'cognito': 'Cognito', 'fargate': 'Fargate', 'elastic beanstalk': 'Elastic Beanstalk',
    'azure': 'Azure', 'google cloud': 'Google Cloud', 'gcp': 'GCP',
    'google cloud run': 'Google Cloud Run', 'firebase': 'Firebase',
    'firestore': 'Firestore', 'cloud functions': 'Cloud Functions',
    'heroku': 'Heroku', 'vercel': 'Vercel', 'netlify': 'Netlify',
    'digitalocean': 'DigitalOcean', 'linode': 'Linode',
    'cloudflare': 'Cloudflare', 'supabase': 'Supabase',
    
    # === DevOps & CI/CD ===
    'docker': 'Docker', 'docker compose': 'Docker Compose',
    'dockerfile': 'Dockerfile', 'kubernetes': 'Kubernetes', 'k8s': 'K8s',
    'helm': 'Helm', 'terraform': 'Terraform', 'ansible': 'Ansible',
    'jenkins': 'Jenkins', 'github actions': 'GitHub Actions',
    'gitlab ci': 'GitLab CI', 'circleci': 'CircleCI',
    'nginx': 'Nginx', 'apache': 'Apache', 'tomcat': 'Tomcat',
    'undertow': 'Undertow', 'caddy': 'Caddy', 'certbot': 'Certbot',
    'grafana': 'Grafana', 'prometheus': 'Prometheus', 'datadog': 'Datadog',
    
    # === Databases ===
    'mysql': 'MySQL', 'postgres': 'PostgreSQL', 'postgresql': 'PostgreSQL',
    'mongodb': 'MongoDB', 'redis': 'Redis', 'elasticsearch': 'Elasticsearch',
    'cassandra': 'Cassandra', 'sqlite': 'SQLite', 'mariadb': 'MariaDB',
    'oracle': 'Oracle', 'sql server': 'SQL Server', 'neo4j': 'Neo4j',
    'couchdb': 'CouchDB', 'influxdb': 'InfluxDB', 'cockroachdb': 'CockroachDB',
    'neon': 'Neon', 'supabase': 'Supabase', 'flyway': 'Flyway',
    'liquibase': 'Liquibase', 'h2': 'H2', 'hikari': 'HikariCP',
    'caffeine': 'Caffeine',
    
    # === Version Control & Collaboration ===
    'git': 'Git', 'github': 'GitHub', 'gitlab': 'GitLab',
    'bitbucket': 'Bitbucket', 'jira': 'Jira', 'confluence': 'Confluence',
    'slack': 'Slack', 'trello': 'Trello', 'notion': 'Notion',
    
    # === OS & Platforms ===
    'linux': 'Linux', 'ubuntu': 'Ubuntu', 'debian': 'Debian',
    'centos': 'CentOS', 'fedora': 'Fedora', 'red hat': 'Red Hat',
    'macos': 'macOS', 'mac os': 'macOS', 'windows': 'Windows',
    'android': 'Android', 'ios': 'iOS', 'chrome os': 'ChromeOS',
    'wsl': 'WSL', 'bash': 'Bash', 'powershell': 'PowerShell',
    'zsh': 'Zsh', 'terminal': 'Terminal',
    
    # === Tools & Editors ===
    'vs code': 'VS Code', 'visual studio code': 'Visual Studio Code',
    'visual studio': 'Visual Studio', 'intellij': 'IntelliJ',
    'android studio': 'Android Studio', 'xcode': 'Xcode',
    'vim': 'Vim', 'neovim': 'Neovim', 'emacs': 'Emacs',
    'sublime': 'Sublime Text', 'atom': 'Atom', 'eclipse': 'Eclipse',
    'postman': 'Postman', 'swagger': 'Swagger', 'insomnia': 'Insomnia',
    'webpack': 'Webpack', 'vite': 'Vite', 'parcel': 'Parcel',
    'babel': 'Babel', 'eslint': 'ESLint', 'prettier': 'Prettier',
    'gradle': 'Gradle', 'maven': 'Maven', 'npm': 'npm', 'yarn': 'Yarn',
    'pnpm': 'pnpm', 'pip': 'pip', 'conda': 'Conda',
    
    # === Protocols & Standards ===
    'http': 'HTTP', 'https': 'HTTPS', 'rest': 'REST', 'restful': 'RESTful',
    'graphql': 'GraphQL', 'grpc': 'gRPC', 'websocket': 'WebSocket',
    'tcp': 'TCP', 'udp': 'UDP', 'ip': 'IP', 'dns': 'DNS',
    'ssl': 'SSL', 'tls': 'TLS', 'ssh': 'SSH', 'ftp': 'FTP',
    'oauth': 'OAuth', 'jwt': 'JWT', 'saml': 'SAML',
    'cors': 'CORS', 'csrf': 'CSRF', 'xss': 'XSS',
    'json': 'JSON', 'xml': 'XML', 'yaml': 'YAML', 'csv': 'CSV',
    'api': 'API', 'sdk': 'SDK', 'cli': 'CLI', 'gui': 'GUI',
    'crud': 'CRUD', 'orm': 'ORM', 'jpa': 'JPA',
    
    # === Concepts & Patterns ===
    'devops': 'DevOps', 'devsecops': 'DevSecOps', 'mlops': 'MLOps',
    'ci cd': 'CI/CD', 'cicd': 'CI/CD',
    'mvvm': 'MVVM', 'mvc': 'MVC', 'mvp': 'MVP',
    'solid': 'SOLID', 'dry': 'DRY', 'kiss': 'KISS',
    'agile': 'Agile', 'scrum': 'Scrum', 'kanban': 'Kanban',
    'microservices': 'Microservices', 'monolith': 'Monolith',
    'serverless': 'Serverless', 'saas': 'SaaS', 'paas': 'PaaS', 'iaas': 'IaaS',
    'rbac': 'RBAC', 'acl': 'ACL',
    
    # === Companies & Products ===
    'google': 'Google', 'microsoft': 'Microsoft', 'apple': 'Apple',
    'amazon': 'Amazon', 'meta': 'Meta', 'facebook': 'Facebook',
    'netflix': 'Netflix', 'uber': 'Uber', 'airbnb': 'Airbnb',
    'spotify': 'Spotify', 'twitter': 'Twitter',
    'openai': 'OpenAI', 'chatgpt': 'ChatGPT',
}


# ============================================================
# STEP 3: TECHNICAL SPEECH-TO-SYMBOL PATTERNS
# ============================================================

# Order matters: longer/more specific patterns FIRST
TECH_SPEECH_PATTERNS = [
    # === Protocol prefixes (must come before individual "colon", "slash") ===
    (r'(?i)\bhttps?\s+colon\s+slash\s+slash\b',
     lambda m: 'https://' if 'https' in m.group().lower() else 'http://'),
    
    # === Domain extensions ===
    (r'(?i)\bdot\s+com\b', '.com'),
    (r'(?i)\bdot\s+in\b', '.in'),
    (r'(?i)\bdot\s+org\b', '.org'),
    (r'(?i)\bdot\s+net\b', '.net'),
    (r'(?i)\bdot\s+io\b', '.io'),
    (r'(?i)\bdot\s+dev\b', '.dev'),
    (r'(?i)\bdot\s+co\b', '.co'),
    (r'(?i)\bdot\s+ai\b', '.ai'),
    (r'(?i)\bdot\s+app\b', '.app'),
    (r'(?i)\bdot\s+cloud\b', '.cloud'),
    
    # === File extensions ===
    (r'(?i)\bdot\s+js\b', '.js'),
    (r'(?i)\bdot\s+ts\b', '.ts'),
    (r'(?i)\bdot\s+py\b', '.py'),
    (r'(?i)\bdot\s+java\b', '.java'),
    (r'(?i)\bdot\s+kt\b', '.kt'),
    (r'(?i)\bdot\s+kts\b', '.kts'),
    (r'(?i)\bdot\s+xml\b', '.xml'),
    (r'(?i)\bdot\s+json\b', '.json'),
    (r'(?i)\bdot\s+yaml\b', '.yaml'),
    (r'(?i)\bdot\s+yml\b', '.yml'),
    (r'(?i)\bdot\s+html\b', '.html'),
    (r'(?i)\bdot\s+css\b', '.css'),
    (r'(?i)\bdot\s+go\b', '.go'),
    (r'(?i)\bdot\s+rb\b', '.rb'),
    (r'(?i)\bdot\s+sh\b', '.sh'),
    (r'(?i)\bdot\s+bat\b', '.bat'),
    (r'(?i)\bdot\s+md\b', '.md'),
    (r'(?i)\bdot\s+txt\b', '.txt'),
    (r'(?i)\bdot\s+env\b', '.env'),
    (r'(?i)\bdot\s+exe\b', '.exe'),
    (r'(?i)\bdot\s+apk\b', '.apk'),
    (r'(?i)\bdot\s+aab\b', '.aab'),
    (r'(?i)\bdot\s+gradle\b', '.gradle'),
    (r'(?i)\bdot\s+sql\b', '.sql'),
    (r'(?i)\bdot\s+csv\b', '.csv'),
    (r'(?i)\bdot\s+pdf\b', '.pdf'),
    (r'(?i)\bdot\s+png\b', '.png'),
    (r'(?i)\bdot\s+jpg\b', '.jpg'),
    (r'(?i)\bdot\s+svg\b', '.svg'),
    (r'(?i)\bdot\s+jar\b', '.jar'),
    (r'(?i)\bdot\s+war\b', '.war'),
    (r'(?i)\bdot\s+properties\b', '.properties'),
    (r'(?i)\bdot\s+toml\b', '.toml'),
    (r'(?i)\bdot\s+dockerfile\b', '.Dockerfile'),
    (r'(?i)\bdot\s+gitignore\b', '.gitignore'),
    (r'(?i)\bdot\s+dockerignore\b', '.dockerignore'),
    (r'(?i)\bdot\s+log\b', '.log'),
    (r'(?i)\bdot\s+cfg\b', '.cfg'),
    (r'(?i)\bdot\s+ini\b', '.ini'),
    (r'(?i)\bdot\s+conf\b', '.conf'),
    
    # === Wildcards & special characters ===
    (r'(?i)\basterisk\b', '*'),
    (r'(?i)\bwild\s*card\b', '*'),
    
    # === Brackets & braces ===
    (r'(?i)\bopen\s+parenthesis\b', '('),
    (r'(?i)\bclose\s+parenthesis\b', ')'),
    (r'(?i)\bopen\s+bracket\b', '['),
    (r'(?i)\bclose\s+bracket\b', ']'),
    (r'(?i)\bopen\s+curly\s*(brace|bracket)?\b', '{'),
    (r'(?i)\bclose\s+curly\s*(brace|bracket)?\b', '}'),
    (r'(?i)\bopen\s+angle\s*bracket\b', '<'),
    (r'(?i)\bclose\s+angle\s*bracket\b', '>'),
    
    # === Comparison & redirect operators ===
    (r'(?i)\bdouble\s+greater\s+than\b', '>>'),
    (r'(?i)\bgreater\s+than\b', '>'),
    (r'(?i)\bless\s+than\b', '<'),
    
    # === Quotes ===
    (r'(?i)\bdouble\s+quote[s]?\b', '"'),
    (r'(?i)\bsingle\s+quote\b', "'"),
    (r'(?i)\bback\s*tick\b', '`'),

    # === Slashes ===
    (r'(?i)\bforward\s+slash\b', '/'),
    (r'(?i)\bback\s*slash\b', '\\'),
    (r'(?i)\bslash\b', '/'),
    
    # === Command-line symbols ===
    (r'(?i)\bdash\s+dash\b', '--'),
    (r'(?i)\bdouble\s+dash\b', '--'),
    (r'(?i)\bdash\b', '-'),
    (r'(?i)\bhyphen\b', '-'),
    
    # === Punctuation symbols ===
    (r'(?i)\bsemicolon\b', ';'),
    (r'(?i)\bsemi\s+colon\b', ';'),
    (r'(?i)\bexclamation\s*(mark|point)?\b', '!'),
    (r'(?i)\bquestion\s+mark\b', '?'),
    
    # === Common symbols ===
    (r'(?i)\bcolon\b', ':'),
    (r'(?i)\bat\s+the\s+rate\b', '@'),
    (r'(?i)\bat\s+sign\b', '@'),
    (r'(?i)\bunderscore\b', '_'),
    (r'(?i)\bhash\s+sign\b', '#'),
    (r'(?i)\bhash\b', '#'),
    (r'(?i)\bdollar\s+sign\b', '$'),
    (r'(?i)\bpercent\s+sign\b', '%'),
    (r'(?i)\bampersand\b', '&'),
    (r'(?i)\bdouble\s+ampersand\b', '&&'),
    (r'(?i)\bdouble\s+pipe\b', '||'),
    (r'(?i)\bpipe\b', '|'),
    (r'(?i)\btilde\b', '~'),
    (r'(?i)\bcaret\b', '^'),
    (r'(?i)\bplus\s+sign\b', '+'),
    (r'(?i)\bdouble\s+equals\b', '=='),
    (r'(?i)\bnot\s+equals?\b', '!='),
    (r'(?i)\bequals\s+to\b', '='),
    (r'(?i)\bequal\s+to\b', '='),
    (r'(?i)\bequals\b', '='),
    (r'(?i)\barrow\b', '->'),
    (r'(?i)\bcomma\b', ','),
]


# ============================================================
# STEP 4: NUMBER VERBALIZATION HANDLER
# ============================================================

WORD_TO_DIGIT = {
    'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
    'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
}

COMPOUND_TENS = {
    'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
    'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
}

WORD_TO_NUMBER = {
    'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
    'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
    'eighteen': '18', 'nineteen': '19',
}

# Words that signal "the next number word is a literal number"
TECH_NUMBER_TRIGGERS = [
    'port', 'version', 'api', 'level', 'sdk', 'build', 'error',
    'status', 'code', 'http', 'response', 'step', 'line', 'run',
    'pid', 'node', 'gradle', 'java', 'python', 'pixel', 'android',
    'ubuntu', 'centos', 'windows', 'macos', 'localhost', 'chapter',
    'episode', 'module', 'lecture', 'session', 'part', 'v',
]


def convert_digit_sequences(text):
    """
    Convert sequences of spoken digits into actual numbers.
    "eight zero eight zero" → "8080"
    "one two seven dot zero dot zero dot one" → "127.0.0.1"
    "port twenty six" → "port 26"
    """
    words = text.split()
    result = []
    i = 0

    while i < len(words):
        word_lower = words[i].lower().rstrip('.,;:!?')
        punctuation = words[i][len(word_lower):]

        # Handle "double zero", "triple zero"
        if word_lower == 'double' and i + 1 < len(words):
            next_lower = words[i+1].lower().rstrip('.,;:!?')
            if next_lower in WORD_TO_DIGIT:
                digit = WORD_TO_DIGIT[next_lower]
                if result and result[-1].isdigit():
                    result[-1] += digit * 2
                else:
                    result.append(digit * 2)
                i += 2
                continue

        if word_lower == 'triple' and i + 1 < len(words):
            next_lower = words[i+1].lower().rstrip('.,;:!?')
            if next_lower in WORD_TO_DIGIT:
                digit = WORD_TO_DIGIT[next_lower]
                if result and result[-1].isdigit():
                    result[-1] += digit * 3
                else:
                    result.append(digit * 3)
                i += 2
                continue

        # Handle compound tens: "twenty six" → "26"
        if word_lower in COMPOUND_TENS:
            tens_val = COMPOUND_TENS[word_lower]
            if i + 1 < len(words):
                next_lower = words[i+1].lower().rstrip('.,;:!?')
                if next_lower in WORD_TO_DIGIT:
                    ones_val = int(WORD_TO_DIGIT[next_lower])
                    num_str = str(tens_val + ones_val)
                    if result and result[-1].isdigit():
                        result[-1] += num_str
                    else:
                        result.append(num_str)
                    i += 2
                    continue
            # Standalone tens: "twenty" → "20"
            prev_is_trigger = (result and
                              result[-1].lower().rstrip('.,;:!?') in TECH_NUMBER_TRIGGERS)
            if prev_is_trigger:
                if result and result[-1].isdigit():
                    result[-1] += str(tens_val)
                else:
                    result.append(str(tens_val))
                i += 1
                continue

        # Handle teens: "fifteen" → "15" (only in tech context)
        if word_lower in WORD_TO_NUMBER:
            prev_is_trigger = (result and
                              result[-1].lower().rstrip('.,;:!?') in TECH_NUMBER_TRIGGERS)
            if prev_is_trigger:
                result.append(WORD_TO_NUMBER[word_lower])
                i += 1
                continue

        # Handle single digits in sequences
        if word_lower in WORD_TO_DIGIT:
            digit = WORD_TO_DIGIT[word_lower]
            if result and result[-1].isdigit():
                result[-1] += digit
                i += 1
                continue
            
            next_is_digit = (i + 1 < len(words) and
                            words[i+1].lower().rstrip('.,;:!?') in WORD_TO_DIGIT)
            next_is_compound = (i + 1 < len(words) and
                               words[i+1].lower().rstrip('.,;:!?') in COMPOUND_TENS)
            prev_is_trigger = (result and
                              result[-1].lower().rstrip('.,;:!?') in TECH_NUMBER_TRIGGERS)

            if next_is_digit or next_is_compound or prev_is_trigger:
                result.append(digit)
                i += 1
                continue

        result.append(words[i])
        i += 1

    return ' '.join(result)


# ============================================================
# STEP 5: TEXT CLEANING PIPELINE
# ============================================================

def fix_proper_nouns(text):
    """Apply proper noun capitalization."""
    # Sort by length (longest first) to prevent partial replacements
    sorted_nouns = sorted(PROPER_NOUNS.items(), key=lambda x: len(x[0]), reverse=True)
    for wrong, correct in sorted_nouns:
        pattern = re.compile(re.escape(wrong), re.IGNORECASE)
        text = pattern.sub(correct, text)
    return text


def apply_tech_speech_patterns(text):
    """Convert verbalized technical symbols to actual characters."""
    result = text

    # Step 1: Convert digit sequences
    result = convert_digit_sequences(result)

    # Step 2: Apply symbol patterns
    for pattern, replacement in TECH_SPEECH_PATTERNS:
        if callable(replacement):
            result = re.sub(pattern, replacement, result)
        else:
            result = re.sub(pattern, replacement, result)

    # Step 3: Clean up spacing around symbols
    result = re.sub(r'  +', ' ', result)
    # Join domain parts: "www .google .com" → "www.google.com"
    result = re.sub(r'\s*\.\s*(?=com|in|org|net|io|dev|co|ai|app|cloud)', '.', result)
    # Join port numbers: "localhost :8080" → "localhost:8080"
    result = re.sub(r'\s*:\s*(?=\d)', ':', result)
    # Fix protocol: "http: //" → "http://"
    result = re.sub(r'(https?)\s*:\s*/\s*/', r'\1://', result)
    # Join email @: "user @gmail" → "user@gmail"
    result = re.sub(r'\s*@\s*', '@', result)
    # Join file extensions: "main .java" → "main.java"
    ext_list = ('js|ts|py|java|kt|kts|xml|json|yaml|yml|html|css|go|rb|sh|bat|'
                'md|txt|env|exe|apk|aab|gradle|sql|csv|pdf|png|jpg|svg|jar|war|'
                'properties|toml|log|cfg|ini|conf|gitignore|dockerignore')
    result = re.sub(rf'\s*\.\s*(?={ext_list})\b', '.', result)

    return result.strip()


def clean_text(text):
    """Full text cleaning pipeline."""
    if not text or not text.strip():
        return ""
    
    # Step 1: Basic cleanup
    text = text.strip()
    
    # Step 2: Proper noun capitalization
    text = fix_proper_nouns(text)
    
    # Step 3: Tech speech-to-symbol conversion
    text = apply_tech_speech_patterns(text)
    
    # Step 4: Fix common Whisper artifacts
    # Remove repeated phrases (anti-hallucination)
    text = re.sub(r'(\b\w+\b)( \1){3,}', r'\1', text)
    
    # Step 5: Ensure proper sentence capitalization
    text = text[0].upper() + text[1:] if text else text
    
    return text


# ============================================================
# STEP 6: PARAGRAPH FORMATTER (TurboScribe-style)
# ============================================================

def format_paragraphs(sentences, words_per_para=45):
    """
    Group sentences into paragraphs of ~45-50 words,
    breaking at sentence boundaries.
    """
    paragraphs = []
    current_para = []
    current_word_count = 0

    for sentence in sentences:
        word_count = len(sentence.split())
        
        if current_word_count + word_count > words_per_para and current_para:
            paragraphs.append(' '.join(current_para))
            current_para = [sentence]
            current_word_count = word_count
        else:
            current_para.append(sentence)
            current_word_count += word_count

    if current_para:
        paragraphs.append(' '.join(current_para))

    return '\n\n'.join(paragraphs)


# ============================================================
# STEP 7: TRANSCRIPTION ENGINE
# ============================================================

def transcribe_file(model, filepath, output_dir=None):
    """
    Transcribe a single audio/video file with timestamps.
    Saves transcript in the same directory as the video (or output_dir).
    Returns the output filename.
    """
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    
    if output_dir:
        output_file = os.path.join(output_dir, f"{name_without_ext}_transcript.txt")
    else:
        file_dir = os.path.dirname(filepath)
        output_file = os.path.join(file_dir, f"{name_without_ext}_transcript.txt") if file_dir else f"{name_without_ext}_transcript.txt"

    print(f"\n{'='*60}")
    print(f"📝 Transcribing: {filepath}")
    print(f"{'='*60}")
    start_time = time.time()

    try:
        segments, info = model.transcribe(
            filepath,
            beam_size=5,
            language="en",
            word_timestamps=True,
            condition_on_previous_text=False,      # Anti-hallucination
            compression_ratio_threshold=2.4,        # Anti-hallucination
            no_speech_threshold=0.6,
            vad_filter=True,                        # Voice Activity Detection
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        all_sentences = []
        current_sentence = []
        last_word_end = 0
        segment_count = 0

        for segment in segments:
            segment_count += 1
            if segment_count % 50 == 0:
                print(f"   ⏳ Processing segment {segment_count}...")

            if not hasattr(segment, 'words') or not segment.words:
                text = clean_text(segment.text)
                if text:
                    all_sentences.append(text)
                continue

            for word in segment.words:
                # Detect pauses > 1.5 seconds (paragraph break opportunity)
                gap = word.start - last_word_end if last_word_end > 0 else 0

                if gap > 1.5 and current_sentence:
                    sentence = clean_text(' '.join(current_sentence))
                    if sentence:
                        all_sentences.append(sentence)
                    current_sentence = []

                current_sentence.append(word.word.strip())
                last_word_end = word.end

        # Flush remaining words
        if current_sentence:
            sentence = clean_text(' '.join(current_sentence))
            if sentence:
                all_sentences.append(sentence)

        # Format into paragraphs
        formatted_text = format_paragraphs(all_sentences)

        # Write output
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Transcript: {filename}\n")
            f.write(f"Path: {filepath}\n")
            f.write(f"Duration: {info.duration:.1f}s ({info.duration/60:.1f} min)\n")
            f.write(f"Language: {info.language} (confidence: {info.language_probability:.1%})\n")
            f.write(f"{'='*60}\n\n")
            f.write(formatted_text)

        elapsed = time.time() - start_time
        ratio = info.duration / elapsed if elapsed > 0 else 0
        print(f"   ✅ Done in {elapsed:.1f}s (speed: {ratio:.1f}x realtime)")
        print(f"   📄 Output: {output_file}")
        print(f"   📊 Segments: {segment_count}, Paragraphs: {formatted_text.count(chr(10)*2) + 1}")

        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_file

    except Exception as e:
        print(f"   ❌ Error transcribing {filename}: {e}")
        return None


# ============================================================
# STEP 8: MAIN EXECUTION
# ============================================================

def main():
    """Main entry point for Google Colab execution."""
    
    print("=" * 60)
    print("🎙️  Technical Video Transcriber v2.1")
    print("   Model: OpenAI Whisper large-v3 (faster-whisper)")
    print("   Features: Recursive subfolder scanning, Tech symbols, numbers, proper nouns")
    print("=" * 60)
    
    SUPPORTED_EXTS = ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.mp3',
                     '.wav', '.m4a', '.flac', '.ogg', '.aac')
    
    input_files = []

    # --- Scan current directory AND subdirectories recursively ---
    print("\n🔍 Scanning current directory and all sub-directories for video/audio files...")
    for root, dirs, files in os.walk('.'):
        # Ignore hidden folders like .git, .ipynb_checkpoints
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTS):
                rel_path = os.path.relpath(os.path.join(root, file), '.')
                input_files.append(rel_path)

    if input_files:
        print(f"   ✅ Found {len(input_files)} file(s) in local folder & sub-directories:")
        for f in input_files[:10]:
            print(f"      → {f}")
        if len(input_files) > 10:
            print(f"      ... and {len(input_files) - 10} more file(s)")
    else:
        # Prompt for Colab upload if no files found locally
        try:
            from google.colab import files as colab_files
            print("\n📁 No files found locally. Upload your video/audio files:")
            uploaded = colab_files.upload()
            input_files = list(uploaded.keys())
            print(f"   Received {len(input_files)} file(s)")
        except ImportError:
            print("❌ No audio/video files found in current directory or any subfolders.")
            return

    # --- Load model ---
    print("\n🔄 Loading Whisper large-v3 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    model = WhisperModel(
        "large-v3",
        device=device,
        compute_type=compute_type,
    )
    print(f"   ✅ Model loaded on {device.upper()} ({compute_type})")

    # --- Transcribe all files ---
    output_files = []
    for filepath in input_files:
        result = transcribe_file(model, filepath)
        if result:
            output_files.append(result)

    # --- Package and download ---
    if not output_files:
        print("\n❌ No transcriptions generated.")
        return

    print(f"\n{'='*60}")
    print(f"✅ Transcription complete! {len(output_files)} file(s) generated.")
    print(f"{'='*60}")

    try:
        from google.colab import files as colab_files
        
        if len(output_files) == 1:
            # Single file — download directly
            print(f"\n📥 Downloading: {output_files[0]}")
            colab_files.download(output_files[0])
        else:
            # Multiple files — zip preserving subfolder structure
            import zipfile
            zip_path = "all_transcripts.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in output_files:
                    # Clean up relative path formatting
                    arcname = os.path.relpath(f, '.') if f.startswith('.') else f
                    zf.write(f, arcname)
            
            print(f"\n📥 Downloading: {zip_path} ({len(output_files)} transcripts preserving subfolder structure)")
            colab_files.download(zip_path)
    except ImportError:
        print("\n📁 Transcripts saved locally:")
        for f in output_files:
            print(f"   → {f}")

    # Final cleanup
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\n🧹 Memory cleaned up. Done!")


# Run if executed directly
if __name__ == "__main__":
    main()

