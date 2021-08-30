(define (problem rovers-3-3-success-0-1-rover0) (:domain rovers-3-3-success-0-1-rover0)
(:objects
    base - waypoint
    quarry - waypoint
    dune - waypoint
    mars - objective
    mothership - lander
    rover0RockStore - store
    rover1soilStore - store
    rover2RockStore - store
    rover2soilStore - store
    rover2cam - camera
    color - mode


	(:private
		rover0 - rover
	)
)
(:init

    ;; private
	(store_of rover0RockStore rover0)
	(equipped_for_rock_analysis rover0)
	(can_traverse rover0 base quarry)
	(can_traverse rover0 quarry base)
	(can_traverse rover0 base dune)
	(can_traverse rover0 dune base)
	(at rover0 base)

	;; public
	(visible quarry base)
	(visible base quarry)
	(visible dune base)
	(visible base dune)
	(visible_from mars dune)
	(at_soil_sample dune)
	(at_rock_sample quarry)
	(at_lander mothership base)
	(empty rover0RockStore)
	(empty rover1soilStore)
	(empty rover2RockStore)
	(empty rover2soilStore)
	(supports rover2cam color)
	(calibration_target rover2cam mars)

)
(:goal
	(and
		(communicated_rock_data quarry)
		(communicated_soil_data dune)
		(communicated_image_data mars color)
	)
)
)