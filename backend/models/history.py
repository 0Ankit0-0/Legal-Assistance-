from datetime import datetime
from bson import ObjectId
from utils.db import get_db
import logging

logger = logging.getLogger(__name__)

class History:
    def __init__(self):
        self.db = get_db()
        self.collection = self.db.history
    
    def save_interaction(self, user_id, interaction_type, input_text, result, model_used, processing_time, file_name=None):
        """Save user interaction to history"""
        try:
            history_data = {
                'user_id': ObjectId(user_id),
                'type': interaction_type,  # 'summary', 'qa', 'compare'
                'input_text': input_text[:1000],  # Truncate long inputs for storage
                'result': result,
                'model_used': model_used,
                'processing_time': processing_time,
                'file_name': file_name,
                'created_at': datetime.utcnow()
            }
            
            result = self.collection.insert_one(history_data)
            logger.info(f"✅ History saved for user: {user_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"❌ Error saving history: {e}")
            return None
    
    def get_user_history(self, user_id, page=1, limit=20, history_type=None):
        """Get paginated user history"""
        try:
            skip = (page - 1) * limit
            query = {'user_id': ObjectId(user_id)}
            
            if history_type:
                query['type'] = history_type
            
            # Get total count for pagination
            total = self.collection.count_documents(query)
            
            # Get paginated results
            history = list(self.collection.find(query)
                          .sort('created_at', -1)
                          .skip(skip)
                          .limit(limit))
            
            # Convert ObjectIds to strings for JSON serialization
            for item in history:
                item['_id'] = str(item['_id'])
                item['user_id'] = str(item['user_id'])
                item['created_at'] = item['created_at'].isoformat()
            
            return {
                'history': history,
                'total': total,
                'page': page,
                'pages': (total + limit - 1) // limit,
                'has_next': page * limit < total,
                'has_prev': page > 1
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting user history: {e}")
            return {
                'history': [],
                'total': 0,
                'page': page,
                'pages': 0,
                'has_next': False,
                'has_prev': False
            }
    
    def get_history_item(self, user_id, history_id):
        """Get specific history item"""
        try:
            item = self.collection.find_one({
                '_id': ObjectId(history_id),
                'user_id': ObjectId(user_id)
            })
            
            if item:
                item['_id'] = str(item['_id'])
                item['user_id'] = str(item['user_id'])
                item['created_at'] = item['created_at'].isoformat()
            
            return item
            
        except Exception as e:
            logger.error(f"❌ Error getting history item: {e}")
            return None
    
    def delete_history_item(self, user_id, history_id):
        """Delete specific history item"""
        try:
            result = self.collection.delete_one({
                '_id': ObjectId(history_id),
                'user_id': ObjectId(user_id)
            })
            
            return result.deleted_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error deleting history item: {e}")
            return False
    
    def clear_user_history(self, user_id):
        """Clear all history for a user"""
        try:
            result = self.collection.delete_many({'user_id': ObjectId(user_id)})
            logger.info(f"✅ Cleared {result.deleted_count} history items for user: {user_id}")
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"❌ Error clearing user history: {e}")
            return 0
    
    def get_usage_stats(self, user_id, days=30):
        """Get usage statistics for user"""
        try:
            from datetime import datetime, timedelta
            
            start_date = datetime.utcnow() - timedelta(days=days)
            
            pipeline = [
                {
                    '$match': {
                        'user_id': ObjectId(user_id),
                        'created_at': {'$gte': start_date}
                    }
                },
                {
                    '$group': {
                        '_id': '$type',
                        'count': {'$sum': 1},
                        'avg_processing_time': {'$avg': '$processing_time'}
                    }
                }
            ]
            
            stats = list(self.collection.aggregate(pipeline))
            
            # Format results
            formatted_stats = {
                'summary': 0,
                'qa': 0,
                'compare': 0,
                'total': 0,
                'avg_processing_time': 0
            }
            
            total_time = 0
            total_requests = 0
            
            for stat in stats:
                interaction_type = stat['_id']
                count = stat['count']
                avg_time = stat['avg_processing_time']
                
                formatted_stats[interaction_type] = count
                formatted_stats['total'] += count
                total_time += avg_time * count
                total_requests += count
            
            if total_requests > 0:
                formatted_stats['avg_processing_time'] = round(total_time / total_requests, 2)
            
            return formatted_stats
            
        except Exception as e:
            logger.error(f"❌ Error getting usage stats: {e}")
            return {
                'summary': 0,
                'qa': 0,
                'compare': 0,
                'total': 0,
                'avg_processing_time': 0
            }