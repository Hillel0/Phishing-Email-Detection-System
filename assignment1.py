import re
import urllib.parse
from typing import Dict, List, Tuple

class PhishingDetector:
    def __init__(self):
        self.urgent_words = {
            'urgent', 'immediately', 'action required', 'account suspended',
            'verify your account', 'suspicious activity'
        }
        
        self.legitimate_domains = {
            'gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com'
        }
    
    def analyze_email(self, email_content: str) -> Dict:
        results = {
            'is_suspicious': False,
            'indicators': [],
            'risk_score': 0
        }
        
        # Check for suspicious URLs
        urls = self._extract_urls(email_content)
        suspicious_urls = self._analyze_urls(urls)
        if suspicious_urls:
            results['indicators'].extend(suspicious_urls)
            results['risk_score'] += len(suspicious_urls) * 25
        
        # Check for urgent language
        urgent_phrases = self._check_urgent_language(email_content)
        if urgent_phrases:
            results['indicators'].extend(urgent_phrases)
            results['risk_score'] += len(urgent_phrases) * 15
        
        # Check for spoofed sender
        sender = self._extract_sender(email_content)
        if sender:
            spoof_indicators = self._check_spoofed_sender(sender)
            if spoof_indicators:
                results['indicators'].extend(spoof_indicators)
                results['risk_score'] += 30
        
        results['is_suspicious'] = results['risk_score'] >= 50
        return results
    
    def _extract_urls(self, content: str) -> List[str]:
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        return re.findall(url_pattern, content)
    
    def _analyze_urls(self, urls: List[str]) -> List[str]:
        suspicious_indicators = []
        for url in urls:
            parsed = urllib.parse.urlparse(url)
            domain = parsed.netloc
            
            # Check for IP address URLs
            if re.match(r'\d+\.\d+\.\d+\.\d+', domain):
                suspicious_indicators.append(f"IP address used in URL: {domain}")
            
            # Check for uncommon TLDs
            if domain.split('.')[-1] not in {'com', 'org', 'edu', 'gov', 'net'}:
                suspicious_indicators.append(f"Uncommon TLD in URL: {domain}")
            
            # Check for suspicious domain patterns
            if any(legit_domain in domain and domain != legit_domain 
                  for legit_domain in self.legitimate_domains):
                suspicious_indicators.append(f"Potential domain spoofing: {domain}")
        
        return suspicious_indicators
    
    def _check_urgent_language(self, content: str) -> List[str]:
        found_phrases = []
        content_lower = content.lower()
        
        for phrase in self.urgent_words:
            if phrase in content_lower:
                found_phrases.append(phrase)
        
        return found_phrases
    
    def _extract_sender(self, content: str) -> str:
        # Simple sender extraction - can be enhanced based on email format
        from_match = re.search(r'From:.*?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', 
                             content, re.IGNORECASE)
        return from_match.group(1) if from_match else None
    
    def _check_spoofed_sender(self, sender: str) -> List[str]:
        indicators = []
        domain = sender.split('@')[1]
        
        # Check for similar domains
        for legit_domain in self.legitimate_domains:
            if domain != legit_domain and self._similar_strings(domain, legit_domain):
                indicators.append(f"Potential spoofed sender domain: {domain}")
        
        return indicators
    
    def _similar_strings(self, str1: str, str2: str) -> bool:
        # Simple string similarity check
        if abs(len(str1) - len(str2)) > 3:
            return False
        
        differences = 0
        for c1, c2 in zip(str1, str2):
            if c1 != c2:
                differences += 1
                if differences > 2:
                    return False
        
        return True


#Example usage
if __name__ == "__main__":
    detector = PhishingDetector()
    
    # Example email content for phishing content
    email_content = """
    From: fake@gmai1.com
    Subject: Urgent Action Required - Account Suspension
    
    Dear User,
    
    Your account has been suspended due to suspicious activity.
    Please verify your account immediately by clicking:
    http://secure-verify.192.168.1.1.com/login
    
    Regards,
    Security Team
    """
    
    # Example email content for safe content
    """
    Subject: Request for a Meeting

    Dear Adam,

    I hope this email finds you well. I would like to schedule a meeting to discuss our progress on the project.

    Would you be available on Sunday or Monday to connect? If these times don't work for you, please let me know your availability, and I'd be happy to adjust accordingly.

    Looking forward to your response.

    Best regards,
    Hillel
    Bears Inc. CEO
    0549405125
    """

    results = detector.analyze_email(email_content)
    print("Analysis Results:")
    print(f"Suspicious: {results['is_suspicious']}")
    print(f"Risk Score: {results['risk_score']}")
    print("\nDetected Indicators:")
    if results:
        print("Urgent language:")
    for indicator in results['indicators']:
        if not indicator.startswith("Potential spoofed sender domain"):
            print(f"- {indicator}")
    for indicator in results['indicators']:
        if indicator.startswith("Potential spoofed sender domain"):
            print(indicator)