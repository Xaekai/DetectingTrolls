import requests
import tweepy
from common import *
def reply_percentages(username, status_id, full_text, model, scaler):
    # send a get request
    tokenized_raveled_padded = tokenize_ravel_pad(final_target)
    scaled = scale_tweet(scaler, tokenize_ravel_padded)
    predicted = predict_scaled_tweet(model, scaled)
    api.update_status(status='@{0} Regular: {1}\nTroll: {2}'.format(username, predicted[0], predicted[1]), in_reply_to_status_id=status_id)

# create a class inheriting from the tweepy  StreamListener
class BotStreamer(tweepy.StreamListener):
    def __init__(self, model, scaler):
        super(BotStreamer, self).__init__(self)
        self.model = model
        self.scaler = scaler
    # Called when a new status arrives which is passed down from the on_data method of the StreamListener
    def on_status(self, status):
        logging.info("Processing a new status " + status.full_text)
        username = status.user.screen_name
        status_id = status.id_str
        full_text = status.full_text
        # entities provide structured data from Tweets including resolved URLs, media, hashtags and mentions without having to parse the text to extract that information
        reply_percentages(username, status_id, full_text, self.model, self.scaler)
        
    def on_error(self, error):
        if status_code == 420:
            logging.error("We are rate limited!")
            return True
