import logging
import os
import sys
import io


class SafeStreamHandler(logging.StreamHandler):
    """
    A StreamHandler that safely encodes Unicode characters on Windows.
    Replaces unsupported characters with ASCII equivalents.
    """
    # Unicode to ASCII replacements for common symbols
    UNICODE_REPLACEMENTS = {
        '\u2713': '[OK]',    # ✓ checkmark
        '\u2717': '[X]',     # ✗ cross
        '\u2714': '[OK]',    # ✔ heavy checkmark
        '\u2716': '[X]',     # ✖ heavy cross
        '\u2022': '*',       # • bullet
        '\u2192': '->',      # → arrow
        '\u2190': '<-',      # ← arrow
        '\u2194': '<->',     # ↔ double arrow
        '\u221a': 'sqrt',    # √ square root
        '\u00b2': '^2',      # ² superscript 2
        '\u00b3': '^3',      # ³ superscript 3
    }
    
    def emit(self, record):
        try:
            msg = self.format(record)
            # Replace known Unicode characters with ASCII equivalents
            for unicode_char, ascii_equiv in self.UNICODE_REPLACEMENTS.items():
                msg = msg.replace(unicode_char, ascii_equiv)
            # Handle any remaining non-ASCII characters
            msg = msg.encode('ascii', errors='replace').decode('ascii')
            stream = self.stream
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(name="HRP_Agent", log_file="hrp_agent.log", level=logging.INFO):
    """
    Sets up a logger that writes to both console and a file.
    Handles Unicode characters properly on Windows.
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Check if handlers already exist to avoid duplicate logs
    if not logger.handlers:
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File Handler - UTF-8 encoding for Unicode support (keeps original characters)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        # Console Handler - Use SafeStreamHandler for Windows compatibility
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

