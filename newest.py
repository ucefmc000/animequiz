import os
import re
import json
import time
import random
import logging
import requests
import numpy as np
from PIL import Image, ImageFilter
from io import BytesIO
import cv2
!pip install deepface
from deepface import DeepFace
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from IPython.display import display, HTML
from difflib import SequenceMatcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('AnimeQuiz')

# Make sure we're using CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Install required packages if not already installed
try:
    import cv2
except ImportError:
    !pip install opencv-python
    import cv2

try:
    from deepface import DeepFace
except ImportError:
    !pip install deepface
    from deepface import DeepFace

class AnimeAPI:
    # ... (other methods) ...

    def generate_character_embeddings(self, anime_id, characters):
        """Generate face embeddings for characters using DeepFace"""
        embeddings_file = os.path.join(self.cache_dir, f"anime_{anime_id}_embeddings.json")

        # Check if cached embeddings exist
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        character_embeddings = []

        for character in characters:
            try:
                # Create a directory for saving character images
                char_img_dir = os.path.join(self.cache_dir, "char_images")
                if not os.path.exists(char_img_dir):
                    os.makedirs(char_img_dir)

                # Download character image
                response = requests.get(character['image_url'])
                response.raise_for_status()

                # Save image temporarily for OpenCV/DeepFace processing
                char_name_safe = re.sub(r'[^a-zA-Z0-9]', '_', character['name'])
                img_path = os.path.join(char_img_dir, f"{char_name_safe}.jpg")

                with open(img_path, 'wb') as f:
                    f.write(response.content)

                # Initialize face_path before the try block
                face_path = None
                try:
                    # Use OpenCV to detect faces
                    image = cv2.imread(img_path)
                    if image is None:
                        logger.warning(f"Failed to load image for {character['name']}")
                        continue

                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                    # Use the Haar cascade classifier for face detection (CPU-based)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                    if len(faces) > 0:
                        # Get the largest face
                        largest_face = max(faces, key=lambda face: face[2] * face[3])
                        x, y, w, h = largest_face

                        # Extract the face region
                        face_img = image[y:y + h, x:x + w]
                        face_path = os.path.join(char_img_dir, f"{char_name_safe}_face.jpg")
                        cv2.imwrite(face_path, face_img)

                        try:
                            # Use DeepFace to get embedding
                            embedding = DeepFace.represent(
                                img_path=face_path,
                                model_name="VGG-Face",
                                enforce_detection=False,
                                detector_backend="opencv"
                            )

                            # Store the face embedding
                            character_embeddings.append({
                                'name': character['name'],
                                'role': character['role'],
                                'embedding': embedding[0]["embedding"] if embedding else None
                            })
                            logger.info(f"Generated embedding for {character['name']}")
                        except Exception as embed_error:
                            logger.warning(f"DeepFace embedding failed for {character['name']}: {embed_error}")
                            character_embeddings.append({
                                'name': character['name'],
                                'role': character['role'],
                                'embedding': None
                            })
                    else:  # Corrected indentation for this else block
                        logger.warning(f"No face detected for character {character['name']}")
                        character_embeddings.append({
                            'name': character['name'],
                            'role': character['role'],
                            'embedding': None
                        })

                    # Clean up temporary files
                    if os.path.exists(img_path):
                        os.remove(img_path)
                    if os.path.exists(face_path):
                        os.remove(face_path)

                    # Be nice to the server
                    time.sleep(1)

                except Exception as e:
                    logger.error(f"Error processing character {character['name']}: {e}")
                    character_embeddings.append({
                        'name': character['name'],
                        'role': character['role'],
                        'embedding': None
                    })

            except Exception as e:
                logger.error(f"Error processing character {character['name']}: {e}")
                character_embeddings.append({
                    'name': character['name'],
                    'role': character['role'],
                    'embedding': None
                })
        return character_embeddings
    # ... (rest of your code) ...

class ScreenshotScraper:
    """Class to scrape screenshots from randomc.net"""
    
    def __init__(self):
        self.base_url = "https://randomc.net"
        self.search_url = "https://randomc.net/?s="
        self.cache_dir = "screenshot_cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def search_anime_articles(self, anime_titles):
        """Search for articles related to an anime using its titles"""
        cache_key = "_".join([title.lower().replace(' ', '_') for title in anime_titles.values() if title])[:100]
        cache_file = os.path.join(self.cache_dir, f"{cache_key}_articles.json")
        
        # Check if cached data exists
        if os.path.exists(cache_file):
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        article_urls = []
        
        # Try each title
        for title_type, title in anime_titles.items():
            if not title:
                continue
                
            try:
                search_query = title.replace(' ', '+')
                full_url = f"{self.search_url}{search_query}"
                
                response = requests.get(full_url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.select('article')
                
                for article in articles:
                    title_tag = article.select_one('.entry-title a')
                    if title_tag:
                        article_url = title_tag['href']
                        article_title = title_tag.text
                        
                        # Check if the article is likely about the anime
                        if self._is_relevant_article(article_title, title):
                            article_urls.append(article_url)
                
                # If we found articles, no need to try other title variations
                if article_urls:
                    break
                    
                # Be nice to the server
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error searching for articles with title '{title}': {e}")
        
        # Cache the article URLs
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(article_urls, f)
        
        return article_urls
    
    def _is_relevant_article(self, article_title, anime_title):
        """Check if an article is relevant to the anime"""
        article_title_lower = article_title.lower()
        anime_title_lower = anime_title.lower()
        
        # Direct match
        if anime_title_lower in article_title_lower:
            return True
            
        # Check for episode mentions
        if re.search(r'episode|ep\.|\bep\b', article_title_lower) and SequenceMatcher(None, article_title_lower, anime_title_lower).ratio() > 0.5:
            return True
            
        return False
    
    def extract_screenshots(self, article_urls, max_articles=5):
        """Extract screenshots from article pages"""
        screenshot_urls = []
        
        # Limit to a reasonable number of articles
        article_urls = article_urls[:max_articles]
        
        for url in article_urls:
            try:
                response = requests.get(url)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                content = soup.select_one('.entry-content')
                
                if not content:
                    continue
                    
                images = content.select('img')
                
                for img in images:
                    img_url = img.get('src', '')
                    
                    # Skip small images, ads, logos, or gifs
                    if not img_url or self._is_non_screenshot(img):
                        continue
                    
                    # Convert relative URL to absolute
                    if not bool(urlparse(img_url).netloc): # Indented here to be part of the loop
                        img_url = urljoin(url, img_url)
                    # Fix URLs that start with '//' (scheme-relative URLs) # Indented here to be part of the loop
                    if img_url.startswith('//'):
                        img_url = 'https:' + img_url
                    
                    screenshot_urls.append(img_url)
                
                # Be nice to the server
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error extracting screenshots from {url}: {e}")
        
        # Filter out duplicates
        screenshot_urls = list(set(screenshot_urls))
        
        return screenshot_urls
    
    def _is_non_screenshot(self, img_tag):
        """Check if an image is likely not a screenshot (ad, logo, icon, etc.)"""
        # Check for small images
        width = img_tag.get('width', '')
        height = img_tag.get('height', '')
        
        try:
            if width and int(width) < 400:
                return True
            if height and int(height) < 225:
                return True
        except ValueError:
            pass
        
        # Check for common non-screenshot class names or alt text
        alt_text = img_tag.get('alt', '').lower()
        img_class = img_tag.get('class', [])
        
        non_screenshot_indicators = ['logo', 'icon', 'banner', 'ad', 'button', 'avatar', 'emoji']
        for indicator in non_screenshot_indicators:
            if (indicator in alt_text) or any(indicator in c for c in img_class):
                return True
        
        # Check for gif images
        src = img_tag.get('src', '')
        if src.lower().endswith('.gif'):
            return True
        
        return False


class ImageAnalyzer:
    """Class to analyze screenshot images and determine difficulty"""

    def __init__(self, character_embeddings):
        self.character_embeddings = character_embeddings
        # Ensure OpenCV face detector is available
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def analyze_image(self, image_url):
        """Analyze an image and return its features"""
        try:
            # Fix URLs that start with '//'
            if image_url.startswith('//'):
                image_url = 'https:' + image_url

            # Create a directory for temporary image files
            temp_dir = "temp_images"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # Download the image
            response = requests.get(image_url)
            response.raise_for_status()

            # Save image temporarily for OpenCV processing
            img_path = os.path.join(
                temp_dir, f"screenshot_{hash(image_url)}.jpg")
            with open(img_path, 'wb') as f:
                f.write(response.content)

            # Load image with OpenCV
            image_cv = cv2.imread(img_path)
            if image_cv is None:
                logger.warning(
                    f"Failed to load image from {image_url}")
                return None

            # Get image dimensions
            height, width = image_cv.shape[:2]

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            face_count = len(faces)

            # Analyze faces and try to match characters
            recognized_characters = []

            if face_count > 0:
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_img = image_cv[y:y + h, x:x + w]
                    face_path = os.path.join(
                        temp_dir, f"face_{hash(image_url)}_{x}_{y}.jpg")
                    cv2.imwrite(face_path, face_img)

                    # Try to identify the character using DeepFace
                    try:
                        face_embedding = DeepFace.represent(
                            img_path=face_path,
                            model_name="VGG-Face",
                            enforce_detection=False,
                            detector_backend="opencv"
                        )

                        if face_embedding:
                            character_matches = []
                            for character in self.character_embeddings:
                                if character['embedding'] is None:
                                    continue

                                # Calculate similarity (cosine distance)
                                try:
                                    char_img_path = os.path.join(
                                        "anime_cache", "char_images", f"{re.sub(r'[^a-zA-Z0-9]', '_', character['name'])}_face.jpg")
                                    # Skip if character image doesn't exist
                                    if not os.path.exists(char_img_path):
                                        continue

                                    result = DeepFace.verify(
                                        img1_path=face_path,
                                        img2_path=char_img_path,
                                        model_name="VGG-Face",
                                        enforce_detection=False,
                                        detector_backend="opencv",
                                        distance_metric="cosine"
                                    )

                                    if result["verified"]:
                                        similarity = 1 - result["distance"]
                                        character_matches.append({
                                            'name': character['name'],
                                            'role': character['role'],
                                            'confidence': similarity
                                        })
                                except Exception as verify_error:
                                    logger.debug(
                                        f"Character verification failed: {verify_error}")
                                    continue  # Continue to the next character if verification fails

                            # Sort matches by confidence and get the best match
                            if character_matches:
                                best_match = sorted(
                                    character_matches, key=lambda x: x['confidence'], reverse=True)[0]
                                recognized_characters.append(best_match)

                    except Exception as face_error:
                        logger.debug(f"Face analysis error: {face_error}")

                    # Clean up temporary face file
                    if os.path.exists(face_path):
                        os.remove(face_path)

            # Calculate visual features

            # Brightness (mean pixel value)
            brightness = np.mean(gray) / 255.0

            # Contrast (standard deviation of pixel values)
            contrast = np.std(gray) / 255.0

            # Edge detection for sharpness/complexity
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.mean(edges) / 255.0

            # Clean up temporary image file
            if os.path.exists(img_path):
                os.remove(img_path)

            # Return analysis results
            return {
                'face_count': face_count,
                'recognized_characters': recognized_characters,
                'brightness': brightness,
                'contrast': contrast,
                'edge_density': edge_density,
                'image_url': image_url
            }

        except Exception as e:
            logger.error(f"Error analyzing image {image_url}: {e}")
            return None

    # ... (rest of the class and other code) ...


    
    def determine_difficulty(self, analysis):
        """Determine the difficulty of a screenshot based on analysis"""
        if not analysis:
            return "Medium"  # Default if analysis failed
        
        # Start with medium difficulty
        difficulty = "Medium"
        
        # Check if a character was recognized
        if analysis['recognized_characters']:
            best_character = sorted(analysis['recognized_characters'], key=lambda x: x['confidence'], reverse=True)[0]
            
            # Use character role as primary difficulty factor
            if best_character['role'] == 'MAIN' and best_character['confidence'] > 0.7:
                difficulty = "Super Easy"
            elif best_character['role'] == 'MAIN':
                difficulty = "Easy"
            elif best_character['role'] == 'SUPPORTING' and best_character['confidence'] > 0.6:
                difficulty = "Easy"
            elif best_character['role'] == 'SUPPORTING':
                difficulty = "Medium"
            else:  # BACKGROUND or low confidence
                difficulty = "Hard"
        else:
            # No recognized characters, use visual features
            # Low brightness, low contrast, or few faces often mean harder screenshots
            visual_score = 0
            
            # More faces = easier
            if analysis['face_count'] >= 3:
                visual_score += 2
            elif analysis['face_count'] > 0:
                visual_score += 1
                
            # Higher brightness = easier
            if analysis['brightness'] > 0.6:
                visual_score += 1
                
            # Higher contrast = easier
            if analysis['contrast'] > 0.2:
                visual_score += 1
                
            # Determine difficulty based on visual score
            if visual_score >= 3:
                difficulty = "Easy"
            elif visual_score == 2:
                difficulty = "Medium"
            else:
                difficulty = "Hard"
        
        return difficulty


class AnimeQuiz:
    """Main class for the anime screenshot quiz"""
    
    def __init__(self):
        self.anime_api = AnimeAPI()
        self.scraper = ScreenshotScraper()
        self.current_anime = None
        self.current_characters = None
        self.character_embeddings = None
        self.screenshots = None
        self.quiz_images = None
    
    def setup_quiz(self, anime_name):
        """Set up a quiz for a given anime"""
        logger.info(f"Setting up quiz for anime: {anime_name}")
        
        # Search for the anime
        self.current_anime = self.anime_api.search_anime(anime_name)
        
        if not self.current_anime:
            return False, "Anime not found. Please try another title."
        
        logger.info(f"Found anime: {self.current_anime['titles']['english'] or self.current_anime['titles']['romaji']}")
        
        # Get characters
        self.current_characters = self.anime_api.get_characters(self.current_anime['id'])
        
        if not self.current_characters:
            return False, "Couldn't find characters for this anime."
        
        logger.info(f"Found {len(self.current_characters)} characters")
        
        # Generate character embeddings
        self.character_embeddings = self.anime_api.generate_character_embeddings(
            self.current_anime['id'],
            self.current_characters
        )
        
        logger.info(f"Generated embeddings for {len([c for c in self.character_embeddings if c['embedding']])} characters")
        
        # Search for articles
        articles = self.scraper.search_anime_articles(self.current_anime['titles'])
        
        if not articles:
            return False, "Couldn't find any articles for this anime on RandomC."
        
        logger.info(f"Found {len(articles)} articles")
        
        # Extract screenshots
        self.screenshots = self.scraper.extract_screenshots(articles)
        
        if not self.screenshots or len(self.screenshots) < 4:
            return False, "Not enough screenshots found for this anime. Please try another title."
        
        logger.info(f"Found {len(self.screenshots)} screenshots")
        
        return True, "Quiz setup successful!"
    
    def generate_quiz(self): # Indentation fixed here
        """Generate a quiz with 4 screenshots of varying difficulty"""
        if not self.screenshots or not self.character_embeddings:
            return False
    
        # Analyze screenshots
        analyzer = ImageAnalyzer(self.character_embeddings)
    
        # Analyze a larger subset of screenshots to ensure we get enough valid ones
        sample_size = min(40, len(self.screenshots))  # Increased from 20 to 40
        sample_screenshots = random.sample(self.screenshots, sample_size)
    
        analyzed_screenshots = []
        for url in sample_screenshots:
            # Fix URLs that start with '//'
            if url.startswith('//'):
                url = 'https:' + url
            
            analysis = analyzer.analyze_image(url)
            if analysis:
                difficulty = analyzer.determine_difficulty(analysis)
                analyzed_screenshots.append({
                    'url': url,
                    'analysis': analysis,
                    'difficulty': difficulty
                })
    
        if len(analyzed_screenshots) < 4:
            logger.warning(f"Not enough valid screenshots found after analysis. Found {len(analyzed_screenshots)} out of 4 needed.")
            # If we don't have enough analyzed screenshots, use random ones
            if len(self.screenshots) >= 4:
                logger.info("Using random screenshots instead")
                random_screenshots = random.sample(self.screenshots, 4)
                quiz_images = []
                for url in random_screenshots:
                    # Fix URLs that start with '//'
                    if url.startswith('//'):
                        url = 'https:' + url
                    quiz_images.append({
                        'url': url,
                        'analysis': None,
                        'difficulty': "Medium"  # Default difficulty
                    })
                self.quiz_images = quiz_images
                return True
            return False
    
    # Rest of the method remains the same...
    
    def display_quiz(self):
        """Display the quiz images using IPython HTML"""
        if not self.quiz_images or len(self.quiz_images) < 4:
            print("Not enough quiz images to display.")
            return
        
        # Define colors for difficulty levels
        difficulty_colors = {
            "Super Easy": "#4CAF50",  # Green
            "Easy": "#2196F3",        # Blue
            "Medium": "#FF9800",      # Orange
            "Hard": "#F44336"         # Red
        }
        
        # Create HTML for quiz display
        html = """
        <style>
            .quiz-container {
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-gap: 10px;
                max-width: 1000px;
            }
            .quiz-image {
                position: relative;
                width: 100%;
            }
            .quiz-image img {
                width: 100%;
                border: 4px solid;
                border-radius: 8px;
            }
            .difficulty-badge {
                position: absolute;
                top: 10px;
                right: 10px;
                padding: 5px 10px;
                color: white;
                border-radius: 4px;
                font-weight: bold;
            }
        </style>
        
        <h2>Anime Screenshot Quiz</h2>
        <p>Can you guess which anime these screenshots are from?</p>
        
        <div class="quiz-container">
        """
        
        for i, img in enumerate(self.quiz_images):
            difficulty = img['difficulty']
            color = difficulty_colors[difficulty]
            
            html += f"""
            <div class="quiz-image">
                <img src="{img['url']}" style="border-color: {color};">
                <div class="difficulty-badge" style="background-color: {color};">{difficulty}</div>
            </div>
            """
        
        html += """
        </div>
        """
        
        display(HTML(html))
    
    def check_answer(self, user_answer):
        """Check if the user's answer matches the anime"""
        if not self.current_anime:
            return False, "No active quiz."
        
        # Get all possible valid titles
        valid_titles = [
            self.current_anime['titles']['romaji'],
            self.current_anime['titles']['english'],
            self.current_anime['titles']['native']
        ]
        
        valid_titles = [title.lower() for title in valid_titles if title]
        
        # Clean up user answer
        user_answer = user_answer.lower().strip()
        
        # Check for exact matches
        if user_answer in valid_titles:
            return True, f"Correct! The anime is {self.current_anime['titles']['romaji']}."
        
        # Check for partial matches with sequence matcher
        for title in valid_titles:
            ratio = SequenceMatcher(None, user_answer, title).ratio()
            if ratio > 0.8:
                return True, f"Correct! The anime is {self.current_anime['titles']['romaji']}."
        
        # Check if the user's answer has the season number wrong but the title is right
        for title in valid_titles:
            # Remove season/part numbers for fuzzy matching
            clean_title = re.sub(r'\s+(season|part|s)\s*\d+', '', title)
            clean_answer = re.sub(r'\s+(season|part|s)\s*\d+', '', user_answer)
            
            ratio = SequenceMatcher(None, clean_answer, clean_title).ratio()
            if ratio > 0.8:
                return True, f"Correct! The anime is {self.current_anime['titles']['romaji']}."
        
        return False, f"Sorry, that's incorrect. The anime is {self.current_anime['titles']['romaji']}."

    def run_quiz(self):
        """Run the interactive quiz"""
        try:
            while True:
                # Get anime title from user
                anime_name = input("\nEnter an anime title to create a quiz: ")
                
                # Set up the quiz
                success, message = self.setup_quiz(anime_name)
                
                if not success:
                    print(message)
                    continue
                
                # Generate quiz
                if not self.generate_quiz():
                    print("Failed to generate quiz. Please try another anime.")
                    continue
                
                # Display quiz
                self.display_quiz()
                
                # Get user's answer
                user_answer = input("\nWhat anime are these screenshots from? ")
                
                # Check answer
                correct, message = self.check_answer(user_answer)
                print(message)
                
                # Ask to play again
                play_again = input("\nDo you want to play again? (yes/no): ")
                if play_again.lower() not in ['yes', 'y']:
                    print("Thanks for playing!")
                    break
                    
        except KeyboardInterrupt:
            print("\nQuiz ended. Thanks for playing!")