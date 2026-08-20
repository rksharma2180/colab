#!/usr/bin/env python3
"""
==============================================================================
 Technical Video Transcriber — Google Colab Edition
 Version: 3.0.0 (Modular & Drive Integrated)
 
 High-accuracy transcription for technical courses using faster-whisper
 with OpenAI large-v3 model. Features:
   - Modular workflow with media_processor.py
   - Smart Bitrate Analyzer & optional MP3 audio extraction
   - Disconnect-Resistant Google Drive integration
   - Automatic cleanup: Deletes original heavy video files after transcription
   - Tech speech-to-symbol conversion (slash→/, dot com→.com, etc.)
   - Number verbalization handling (eight zero eight zero → 8080)
   - Proper noun capitalization (Docker, Kubernetes, Linux, etc.)
   - TurboScribe-style paragraph formatting
==============================================================================
"""

import os
import re
import gc
import time
import shutil
import glob
import zipfile
from pathlib import Path

# Default processing mode if not specified: 'audio' (Extract MP3 64k mono), 'video' (Smart x265 compression), or 'none'
if 'PROCESSING_MODE' not in globals():
    PROCESSING_MODE = "audio"

# Import modular media processor if available
try:
    import media_processor
except ImportError:
    media_processor = None


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
            print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
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
    'angularjs': 'AngularJS', 'vue': 'Vue', 'vue.js': 'Vue.js', 'next.js': 'Next.js',
    'nextjs': 'Next.js', 'nuxt': 'Nuxt', 'svelte': 'Svelte', 'django': 'Django',
    'flask': 'Flask', 'fastapi': 'FastAPI', 'express': 'Express', 'express.js': 'Express.js',
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
    
    # === Java & Collections Framework ===
    'arraylist': 'ArrayList', 'linkedlist': 'LinkedList', 'hashset': 'HashSet',
    'treeset': 'TreeSet', 'hashmap': 'HashMap', 'treemap': 'TreeMap',
    'linkedhashmap': 'LinkedHashMap', 'concurrenthashmap': 'ConcurrentHashMap',
    'vector': 'Vector', 'stack': 'Stack', 'priorityqueue': 'PriorityQueue',
    'nullpointerexception': 'NullPointerException', 'classcastexception': 'ClassCastException',
    'arrayindexoutofboundsexception': 'ArrayIndexOutOfBoundsException',
    'illegalargumentexception': 'IllegalArgumentException',
    'system.out.println': 'System.out.println', 'stringbuilder': 'StringBuilder',
    'stringbuffer': 'StringBuffer', 'comparable': 'Comparable', 'comparator': 'Comparator',
    'iterator': 'Iterator', 'listiterator': 'ListIterator', 'enumeration': 'Enumeration',
    
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
    'crud': 'CRUD', 'orm': 'ORM', 'jpa': 'JPA', 'jvm': 'JVM',
    'jdk': 'JDK', 'jre': 'JRE',
    
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

TECH_SPEECH_PATTERNS = [
    (r'(?i)\bhttps?\s+colon\s+slash\s+slash\b',
     lambda m: 'https://' if 'https' in m.group().lower() else 'http://'),
    
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
    
    (r'(?i)\basterisk\b', '*'),
    (r'(?i)\bwild\s*card\b', '*'),
    
    (r'(?i)\bopen\s+parenthesis\b', '('),
    (r'(?i)\bclose\s+parenthesis\b', ')'),
    (r'(?i)\bopen\s+bracket\b', '['),
    (r'(?i)\bclose\s+bracket\b', ']'),
    (r'(?i)\bopen\s+curly\s*(brace|bracket)?\b', '{'),
    (r'(?i)\bclose\s+curly\s*(brace|bracket)?\b', '}'),
    (r'(?i)\bopen\s+angle\s*bracket\b', '<'),
    (r'(?i)\bclose\s+angle\s*bracket\b', '>'),
    
    (r'(?i)\bdouble\s+greater\s+than\b', '>>'),
    (r'(?i)\bgreater\s+than\b', '>'),
    (r'(?i)\bless\s+than\b', '<'),
    
    (r'(?i)\bdouble\s+quote[s]?\b', '"'),
    (r'(?i)\bsingle\s+quote\b', "'"),
    (r'(?i)\bback\s*tick\b', '`'),

    (r'(?i)\bforward\s+slash\b', '/'),
    (r'(?i)\bback\s*slash\b', lambda m: '\\'),
    (r'(?i)\bslash\b', '/'),
    
    (r'(?i)\bdash\s+dash\b', '--'),
    (r'(?i)\bdouble\s+dash\b', '--'),
    (r'(?i)\bdash\b', '-'),
    (r'(?i)\bhyphen\b', '-'),
    
    (r'(?i)\bsemicolon\b', ';'),
    (r'(?i)\bsemi\s+colon\b', ';'),
    (r'(?i)\bexclamation\s*(mark|point)?\b', '!'),
    (r'(?i)\bquestion\s+mark\b', '?'),
    
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

TECH_NUMBER_TRIGGERS = [
    'port', 'version', 'api', 'level', 'sdk', 'build', 'error',
    'status', 'code', 'http', 'response', 'step', 'line', 'run',
    'pid', 'node', 'gradle', 'java', 'python', 'pixel', 'android',
    'ubuntu', 'centos', 'windows', 'macos', 'localhost', 'chapter',
    'episode', 'module', 'lecture', 'session', 'part', 'v',
]


def convert_digit_sequences(text):
    words = text.split()
    result = []
    i = 0

    while i < len(words):
        word_lower = words[i].lower().rstrip('.,;:!?')

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
            prev_is_trigger = (result and
                              result[-1].lower().rstrip('.,;:!?') in TECH_NUMBER_TRIGGERS)
            if prev_is_trigger:
                if result and result[-1].isdigit():
                    result[-1] += str(tens_val)
                else:
                    result.append(str(tens_val))
                i += 1
                continue

        if word_lower in WORD_TO_NUMBER:
            prev_is_trigger = (result and
                              result[-1].lower().rstrip('.,;:!?') in TECH_NUMBER_TRIGGERS)
            if prev_is_trigger:
                result.append(WORD_TO_NUMBER[word_lower])
                i += 1
                continue

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
    sorted_nouns = sorted(PROPER_NOUNS.items(), key=lambda x: len(x[0]), reverse=True)
    for wrong, correct in sorted_nouns:
        pattern = re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE)
        text = pattern.sub(correct, text)
    return text


def apply_tech_speech_patterns(text):
    result = text
    result = convert_digit_sequences(result)

    for pattern, replacement in TECH_SPEECH_PATTERNS:
        if callable(replacement):
            result = re.sub(pattern, replacement, result)
        else:
            result = re.sub(pattern, replacement, result)

    result = re.sub(r'  +', ' ', result)
    result = re.sub(r'\s*\.\s*(?=com|in|org|net|io|dev|co|ai|app|cloud)', '.', result)
    result = re.sub(r'\s*:\s*(?=\d)', ':', result)
    result = re.sub(r'(https?)\s*:\s*/\s*/', r'\1://', result)
    result = re.sub(r'\s*@\s*', '@', result)
    ext_list = ('js|ts|py|java|kt|kts|xml|json|yaml|yml|html|css|go|rb|sh|bat|'
                'md|txt|env|exe|apk|aab|gradle|sql|csv|pdf|png|jpg|svg|jar|war|'
                'properties|toml|log|cfg|ini|conf|gitignore|dockerignore')
    result = re.sub(rf'\s*\.\s*(?={ext_list})\b', '.', result)

    return result.strip()


def clean_text(text):
    if not text or not text.strip():
        return ""
    text = text.strip()
    text = fix_proper_nouns(text)
    text = apply_tech_speech_patterns(text)
    text = re.sub(r'(\b\w+\b)( \1){3,}', r'\1', text)
    text = text[0].upper() + text[1:] if text else text
    return text


def format_paragraphs(sentences, words_per_para=45):
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
# STEP 6: TRANSCRIPTION ENGINE
# ============================================================

def transcribe_file(model, filepath, transcript_dir):
    """
    Transcribes media file and saves transcript into transcript_dir.
    """
    filename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(filename)[0]
    output_file = os.path.join(transcript_dir, f"{name_without_ext}_transcript.txt")

    print(f"\n{'='*60}")
    print(f"📝 Transcribing: {filename}")
    print(f"{'='*60}")
    start_time = time.time()

    try:
        segments, info = model.transcribe(
            filepath,
            beam_size=5,
            language="en",
            word_timestamps=True,
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
            vad_filter=True,
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
                gap = word.start - last_word_end if last_word_end > 0 else 0

                if gap > 1.5 and current_sentence:
                    sentence = clean_text(' '.join(current_sentence))
                    if sentence:
                        all_sentences.append(sentence)
                    current_sentence = []

                current_sentence.append(word.word.strip())
                last_word_end = word.end

        if current_sentence:
            sentence = clean_text(' '.join(current_sentence))
            if sentence:
                all_sentences.append(sentence)

        formatted_text = format_paragraphs(all_sentences)

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Transcript: {filename}\n")
            f.write(f"Duration: {info.duration:.1f}s ({info.duration/60:.1f} min)\n")
            f.write(f"Language: {info.language} (confidence: {info.language_probability:.1%})\n")
            f.write(f"{'='*60}\n\n")
            f.write(formatted_text)

        elapsed = time.time() - start_time
        ratio = info.duration / elapsed if elapsed > 0 else 0
        print(f"   ✅ Done in {elapsed:.1f}s (speed: {ratio:.1f}x realtime)")
        print(f"   📄 Output Transcript: {output_file}")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return output_file

    except Exception as e:
        print(f"   ❌ Error transcribing {filename}: {e}")
        return None


# ============================================================
# STEP 7: MAIN EXECUTION & DRIVE SETUP
# ============================================================

def main():
    print("=" * 60)
    print("🎙️  Technical Video Transcriber v3.0 (Modular & Drive Integrated)")
    print("   Model: OpenAI Whisper large-v3 (faster-whisper)")
    print("   Features: Pre-compression, Drive Sync, Auto Cleanup")
    print("=" * 60)

    # Detect Google Drive environment
    is_colab = False
    drive_root = "."
    try:
        from google.colab import drive
        is_colab = True
        drive_root = "/content/drive/MyDrive/Colab_Transcriber"
        print(f"   📁 Google Drive root: {drive_root}")
    except ImportError:
        drive_root = "./Colab_Transcriber"
        print(f"   📁 Local execution root: {drive_root}")

    # Directories setup
    orig_dir = os.path.join(drive_root, "original")
    comp_dir = os.path.join(drive_root, "compressed")
    txt_dir = os.path.join(drive_root, "transcripts")

    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    SUPPORTED_EXTS = ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.mp3',
                     '.wav', '.m4a', '.flac', '.ogg', '.aac')

    # Step 1: Scan for original files in original/ directory
    original_files = []
    for root, dirs, files in os.walk(orig_dir):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for file in files:
            if file.lower().endswith(SUPPORTED_EXTS):
                original_files.append(os.path.join(root, file))

    # Processing Mode configuration (Supports %run transcriber.py video / audio / none)
    import sys
    proc_mode = "audio"
    if len(sys.argv) > 1:
        arg_mode = sys.argv[1].lower().replace('--mode=', '').replace('-', '')
        if arg_mode in ('audio', 'video', 'none'):
            proc_mode = arg_mode
    elif 'PROCESSING_MODE' in globals():
        proc_mode = str(globals()['PROCESSING_MODE']).lower()
    elif 'PROCESSING_MODE' in os.environ:
        proc_mode = os.environ['PROCESSING_MODE'].lower()

    print(f"\n🎛️ Processing Mode: {proc_mode.upper()}")

    if not original_files:
        print(f"\n❌ No video/audio files found in Google Drive:")
        print(f"   👉 Please place your video files into: {os.path.abspath(orig_dir)}")
        print(f"   Then run `%run transcriber.py video` again!")
        return

    if not original_files:
        print("❌ No input files to process.")
        return

    # Step 2: Pre-process files (Compress / Extract) BEFORE loading Whisper
    print(f"\n⚙️ Pre-Processing {len(original_files)} file(s) [Mode: {proc_mode.upper()}]...")
    processed_files = []
    
    for orig_path in original_files:
        rel_path = os.path.relpath(orig_path, orig_dir)
        target_sub_dir = os.path.join(comp_dir, os.path.dirname(rel_path))
        
        # Check if transcript already exists in Drive -> skip preprocessing!
        name_no_ext = os.path.splitext(os.path.basename(orig_path))[0]
        expected_txt = os.path.join(txt_dir, os.path.dirname(rel_path), f"{name_no_ext}_transcript.txt")
        
        if os.path.exists(expected_txt) and os.path.getsize(expected_txt) > 0:
            print(f"   ⏩ Transcript already exists: {expected_txt} (Skipping preprocessing)")
            processed_files.append((orig_path, None, expected_txt))
            continue

        if media_processor:
            proc_path = media_processor.process_media_file(orig_path, target_sub_dir, mode=proc_mode)
        else:
            proc_path = orig_path

        processed_files.append((orig_path, proc_path, expected_txt))

    # Step 3: Load Whisper model into GPU memory
    print("\n🔄 Loading Whisper large-v3 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    model = WhisperModel(
        "large-v3",
        device=device,
        compute_type=compute_type,
    )
    print(f"   ✅ Model loaded on {device.upper()} ({compute_type})")

    # Step 4: Transcribe & Drive Cleanup
    output_transcripts = []
    for orig_path, proc_path, expected_txt in processed_files:
        # Check if already completed
        if os.path.exists(expected_txt) and os.path.getsize(expected_txt) > 0:
            output_transcripts.append(expected_txt)
            # Cleanup heavy original if still present
            if os.path.exists(orig_path):
                os.remove(orig_path)
                print(f"   🧹 Drive Cleanup: Deleted heavy original video {orig_path}")
            continue

        if proc_path and os.path.exists(proc_path):
            txt_out_dir = os.path.dirname(expected_txt)
            res = transcribe_file(model, proc_path, txt_out_dir)
            if res:
                output_transcripts.append(res)
                # Cleanup heavy original video from Drive to save space!
                if os.path.exists(orig_path):
                    try:
                        os.remove(orig_path)
                        print(f"   🧹 Drive Cleanup: Deleted original video {os.path.basename(orig_path)}")
                    except Exception as e:
                        print(f"   ⚠️ Cleanup warning: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"✅ Finished! {len(output_transcripts)} transcript(s) stored safely in Google Drive:")
    print(f"   📁 Transcripts: {txt_dir}")
    print(f"   📁 Processed Media: {comp_dir}")
    print(f"{'='*60}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
