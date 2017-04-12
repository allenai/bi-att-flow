/**
 * Get a random floating point number between `min` and `max`.
 * 
 * @param {number} min - min number
 * @param {number} max - max number
 * @return {float} a random floating point number
 */
function getRandom(min, max) {
  return Math.random() * (max - min) + min;
}

/**
 * Get a random integer between `min` and `max`.
 * 
 * @param {number} min - min number
 * @param {number} max - max number
 * @return {int} a random integer
 */
function getRandomInt(min, max) {
  return Math.floor(Math.random() * (max - min + 1) + min);
}

jQuery(document).ready(function($) {
	// Load contexts and questions from layout
	var contexts = jQuery.parseJSON($('#contexts_json').val());
	var contexts_questions = jQuery.parseJSON($('#context_questions_json').val());
	console.log(contexts_questions);
	console.log(contexts)	

	// Set default paramenters
	$('#context').text(contexts[0][0]);
	question = contexts_questions['0']['0'][getRandomInt(0,contexts_questions['0']['0'].length)]
	$('#question').prop('value', question.join(' '));

	// Load new context when select changes
	$('#selectArticle').change(function(item){
		article_idx = $(this).find("option:selected").data('article-idx');
		paragraph_idx = $(this).find("option:selected").data('paragraph-idx');
		$('#context').text(contexts[article_idx][paragraph_idx]);
		question = contexts_questions[article_idx][paragraph_idx][getRandomInt(0,contexts_questions[article_idx][paragraph_idx].length)]
		$('#question').prop('value', question.join(' '));
	})

	$('#random_context_question').click(function(){
		event.preventDefault();
		article_idx = $('#selectArticle').find("option:selected").data('article-idx');
		paragraph_idx = $('#selectArticle').find("option:selected").data('paragraph-idx');

		question = contexts_questions[article_idx][paragraph_idx][getRandomInt(0,contexts_questions[article_idx][paragraph_idx].length)]
		$('#question').prop('value', question.join(' '));
	})

	$('#ask_question').submit(function(event){
		
		// console.log($(this).serialize());		
		$.ajax({
			url: "http://0.0.0.0:1995/submit",
			type: "POST",
			dataType: "json",
			context: document.body,	
			data: {
				'context': $('#context').val(),
				'question': $('#question').val()
			},
			beforeSend: function(xhr) {
				$('#answer').prop('value','...loading answer');
			},
			success: function(data) {
				// console.log(data);
				$('#answer').prop('value',data.answers[0]);
				$('#answer_confidence').prop('value',data.scores[0]);
				
				$('#alternative_answers li').remove();
				for (var i = 1; i < data.answers.length; i++) {
					$('#alternative_answers').append('<li>'+data.answers[i]+' <em>Confidence ('+data.scores[i]+')</em></li>');
				}
			},
			error: function(data) {
				$('#answer').prop('value','Sorry, we could not get an answer.');
			}
		});
		return false;
	})
})